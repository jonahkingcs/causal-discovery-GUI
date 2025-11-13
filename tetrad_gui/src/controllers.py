# Qt widgets for opening files and pop ups
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QCursor
# py-tetrad wrapper
from tetrad_iface import TetradRunner
# matplotlib canvas class for displaying figures in a Qt widget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
import networkx as nx
from dialogs import GfciParamDialog, FgesParamDialog, ImportWizard, GfciWizard, PcWizard, FgesWizard, TutorialsHub, EdgeExplanationDialog
import math
import os
import re
import html
import logging
import time
# logs for unit testing
log = logging.getLogger("tetrad_gui")
from matplotlib.transforms import Bbox
try:
    # if graphviz is installed a nicer layout is used
    from networkx.drawing.nx_pydot import graphviz_layout
    _HAS_GRAPHVIZ = True
except Exception:
    _HAS_GRAPHVIZ = False



# edge explanation templates
_EDGE_TEMPLATES = {
    "-->": """<span style="color:#003366; font-size:18pt; font-weight:bold;">A --&gt; B</span><br><br>
<b>Meaning:</b> A is a cause of B. This could be a direct link or indirect via mediators. There could also be an unmeasured common cause in the background affecting both A and B. This rules out the idea that B causes A.""",

    "<->": """<span style="color:#003366; font-size:18pt; font-weight:bold;">A &lt;-&gt; B</span><br><br>
<b>Meaning:</b> Neither A nor B cause each other directly. Association between them is explained best by an unmeasured confounder influencing both. A hidden variable X with relationships X --&gt; A and X --&gt; B could exist.""",

    "o->": """<span style="color:#003366; font-size:18pt; font-weight:bold;">A o-&gt; B</span><br><br>
<b>Meaning:</b> The arrow at B is certain (B is an effect), but we are unsure at A. Either A causes B or an unmeasured confounder of A and B exists or both. Either A --&gt; B, A &lt;-&gt; B, A --&gt; B with an extra confounder could be the case.""",

    "o-o": """<span style="color:#003366; font-size:18pt; font-weight:bold;">A o-o B</span><br><br>
<b>Meaning:</b> From the data and assumptions used, we cannot tell whether it’s A --&gt; B or B --&gt; A, a hidden confounder, or a combination. Any of these may be true""",

    "---": """<span style="color:#003366; font-size:18pt; font-weight:bold;">A --- B</span><br><br>
<b>Meaning:</b> they are adjacent, but the direction is uncertain. This is how a CPDAG tells us there is a relationship, but it cannot decide between A --&gt; B or B --&gt; A, purely from observational data. It omits the existence of a hidden confounding variable."""
}



# Accept common display variants and normalize to our template keys
_MARK_NORMALIZE = {
    "->":  "-->",
    "→":   "-->",
    "-->": "-->",
    "<->": "<->",
    "↔":   "<->",
    "o->": "o->",
    "o→":  "o->",
    "o-o": "o-o",
    "---": "---",
    "—":   "---",
    "-":   "---",   # if your list ever shows a single dash for undirected
}



def _format_edge_explanation(template: str, A: str, B: str) -> str:
    A_ = html.escape(A); B_ = html.escape(B)
    s = template

    # replace explicit edge tokens
    for old, new in [
        ("A --> B", f"{A} --> {B}"),   ("B --> A", f"{B} --> {A}"),
        ("A --&gt; B", f"{A_} --&gt; {B_}"), ("B --&gt; A", f"{B_} --&gt; {A_}"),
        ("A <-> B", f"{A} <-> {B}"),   ("B <-> A", f"{B} <-> {A}"),
        ("A &lt;-&gt; B", f"{A_} &lt;-&gt; {B_}"), ("B &lt;-&gt; A", f"{B_} &lt;-&gt; {A_}"),
        ("A o-> B", f"{A} o-> {B}"),   ("B o-> A", f"{B} o-> {A}"),
        ("A o-&gt; B", f"{A_} o-&gt; {B_}"), ("B o-&gt; A", f"{B_} o-&gt; {A_}"),
        ("A o-o B", f"{A} o-o {B}"),   ("B o-o A", f"{B} o-o {A}"),
        ("A --- B", f"{A} --- {B}"),   ("B --- A", f"{B} --- {A}"),
    ]:
        s = s.replace(old, new)

    # replace A and B with variable names
    for old, new in [
        ("B causes A", f"{B_} causes {A_}"),
        ("A causes B", f"{A_} causes {B_}"),
        ("affecting both A and B", f"affecting both {A_} and {B_}"),
        ("of A and B exists", f"of {A_} and {B_} exists"),
        ("of A and B", f"of {A_} and {B_}"),
        ("both A and B", f"both {A_} and {B_}"),
        ("Neither A nor B", f"Neither {A_} nor {B_}"),
        ("at B", f"at {B_}"),
        ("at A", f"at {A_}"),
        ("(B is an effect)", f"({B_} is an effect)"),
        ("X --> A", f"X --&gt; {A_}"),
        ("X --> B", f"X --&gt; {B_}"),
        ("A is a cause of B", f"{A_} is a cause of {B_}"),
    ]:
        s = s.replace(old, new)

    return s



# parses text string into correct edge labels
def _parse_edge_label(text: str):
    m = re.match(r'^(.*?)\s+(-->|->|<->|o->|o-o|---)\s+(.*?)$', text)
    if not m:
        return None
    A, mark, B = m.group(1).strip(), m.group(2), m.group(3).strip()
    if mark == "->":
        mark = "-->"
    return A, mark, B



class Controller:
    def __init__(self, ui):
        # references MainWindow loaded from the .ui file
        self.ui = ui
        # holds loaded pandas dataframe
        self.df = None
        # TetradRunner instance that can load data and run the search algorithms
        self.tetrad = TetradRunner()

        self.data_filename = None 

        # graph title
        self.graph_title = ""
        # default significance level
        self.alpha = 0.05
        # default depth
        self.depth = -1
        # data type (discrete or continuous)
        self.dtype = "auto"
        # default penalty for FGES
        self.penalty_discount = 2.0

        # node (x, y)
        self._pos           = {}
        self._node_artists  = {}
        self._edge_artists  = []
        # name of node being dragged
        self._dragging_node = None

        # Matplotlib figure
        self._figure = None
        self._canvas = None

        # memoises learned graphs per algorithm
        self._result_cache = {}
        # memoises comparison text
        self._compare_cache = {}
        # tracks last generated comparison signature
        self._last_compare_key = None

        # displayed graph
        self._current_graph = None



    # loads data triggered by open CSV tool
    def load_data(self):
        # show tutorial
        wiz = ImportWizard(self.ui)
        # when user cancels
        if wiz.exec_() != QtWidgets.QDialog.Accepted:
            return

        path, _ = QFileDialog.getOpenFileName(
            self.ui, "Open CSV", "", "CSV Files (*.csv)"
        )
        # if user cancels
        if not path:
            return
        # reads CSV file into a pandas dataframe using the TetradRunner
        self.df = self.tetrad.load_dataframe(path)

        self.data_filename = os.path.basename(path)

        problems = self._validate_df(self.df)
        if problems:
            QtWidgets.QMessageBox.warning(
                self.ui, "CSV not usable",
                "Please fix the following issues and re-import:\n\n• " + "\n• ".join(problems)
            )
            self.df = None
            self.ui.statusBar().clearMessage()
            return

        self._invalidate_caches()

        # status bar feedback
        self.ui.statusBar().showMessage(f"Loaded {len(self.df)} rows")
       


    # handles clicking on the tools list at the side
    def handle_tool_click(self, item):
        # the clicked rows label
        text = item.text()

        # Import Data tool clicked
        if text == "Import Data":
            self.load_data()
            return

        if text == "GFCI Search":

            if self.df is None:
                QMessageBox.warning(self.ui, "No data", "Load data first (Tools -> Import Data).")
                return
            
            # open the GFCI tutorial
            settings = QSettings("MyUni", "CausalGUI")
            if settings.value("show_gfci_wizard", True, type=bool):
                wiz = GfciWizard(self.ui)
                if wiz.exec_() != QtWidgets.QDialog.Accepted:
                    # when user hits cancel
                    return
                if wiz.dontShowAgain():
                    settings.setValue("show_gfci_wizard", False)

            # open the parameter dialog
            dlg = GfciParamDialog(self.ui, self.alpha, self.depth, self.dtype, initial_title=self.graph_title)
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                # when user hits cancel
                return

            # pull chosen values and remember them
            self.alpha, self.depth, self.dtype, self.graph_title = dlg.values()

            self._invalidate_caches()

            # run with those parameters
            self.run_algorithm("GFCI")
            return
        
        if text == "PC Search":

            if self.df is None:
                QMessageBox.warning(self.ui, "No data",
                                    "Load data first (Tools -> Import Data).")
                return
            
            # PC tutorial wizard
            settings = QSettings("MyUni", "CausalGUI")
            if settings.value("show_pc_wizard", True, type=bool):
                wiz = PcWizard(self.ui)
                if wiz.exec_() != QtWidgets.QDialog.Accepted:
                    return
                if wiz.dontShowAgain():
                    settings.setValue("show_pc_wizard", False)
            
            # open the parameter dialog
            dlg = GfciParamDialog(self.ui, self.alpha, self.depth, self.dtype, initial_title=self.graph_title)
            dlg.setWindowTitle("Run PC: parameters")           
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return

            # pull chosen values and remember them
            self.alpha, self.depth, self.dtype, self.graph_title = dlg.values()

            self._invalidate_caches()

            # run with those parameters
            self.run_algorithm("PC")
            return
        
        if text == "FGES Search":
            if self.df is None:
                QMessageBox.warning(self.ui, "No data", "Load data first (Tools -> Import Data).")
                return
            
            # FGES tutorial wizard
            settings = QSettings("MyUni", "CausalGUI")
            if settings.value("show_fges_wizard", True, type=bool):
                from dialogs import FgesWizard
                wiz = FgesWizard(self.ui)
                if wiz.exec_() != QtWidgets.QDialog.Accepted:
                    return
                if wiz.dontShowAgain():
                    settings.setValue("show_fges_wizard", False)
                    
            # open the parameter dialog
            dlg = FgesParamDialog(self.ui, penalty=self.penalty_discount, dtype=self.dtype, initial_title=self.graph_title)
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
            
            # pull chosen values and remember them
            self.penalty_discount, self.dtype, self.graph_title = dlg.values()

            self._invalidate_caches()

            # run with those parameters
            self.run_algorithm("FGES")
            return

        # any Search tool clicked
        if text.endswith("Search"):
            algo = text.split()[0].upper()

            # check data is loaded
            if self.df is None:
                QMessageBox.warning(
                    self.ui,
                    "No data",
                    "Please load data first (Tools ➜ Import Data)."
                )
                return

            # ask to confirm search
            reply = QMessageBox.question(
                self.ui,
                f"Run {algo}?",
                f"Are you sure you want to run a {algo} search on the loaded data?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.run_algorithm(algo)
            return

        # for tools not implemented yet
        QMessageBox.information(
            self.ui,
            "Not Implemented",
            f"The tool “{text}” isn't wired yet."
        )



    # run search algorithms
    def run_algorithm(self, algo):
        # stops running an algorithm with no loaded data
        if self.df is None:
            QMessageBox.warning(self.ui, "No data", "Load data first.")
            return
        
        # tells user which algorithm is running in status bar
        self.ui.statusBar().showMessage(f"Running {algo} ...")


        t0 = time.perf_counter()

        try:
            # passes dataframe and algorithm to the TetradRunner and returns a networkx graph
            nx_graph = self.tetrad.run_search(self.df, algo, alpha=self.alpha, depth=self.depth, dtype=self.dtype, penalty_discount=getattr(self, "penalty_discount", 2.0))
        except Exception as e:
            # build a message to map common backend errors to guidance
            emsg = str(e)
            hints = [
                "• The CSV must have a header row of variable names (no blank/duplicate names).",
                "• Choose a matching data type in the parameter dialog:",
                "    - Continuous: numeric columns only (floats/ints).",
                "    - Discrete: small integer codes (0,1,2,...) for categories.",
                "• Remove text like 'NA' in numeric columns.",
            ]
            if "Data set must be continuous" in emsg:
                hints.insert(0, "Data type mismatch: you selected 'Continuous' but the data isn't purely numeric.")
            QtWidgets.QMessageBox.critical(
                self.ui,
                f"{algo} could not run",
                "The search failed on this dataset.\n\n"
                + "\n".join(hints)
            )
            self.ui.statusBar().clearMessage()
            return
        
        dt = time.perf_counter() - t0
        rows = len(self.df)
        cols = len(getattr(self.df, "columns", []))
        print(f"PERF run_{algo}_s={dt:.3f}  shape={rows}x{cols}  dtype={self.dtype}")
        
        # prints directed edges in the console for debugging purposes
        for i, (u, v) in enumerate(nx_graph.edges, start=1):
            print(f"{i}. {u} --> {v}")

        # renders graph into the central canvas
        self.draw_graph(nx_graph, algo, custom_title=self.graph_title)
        self._current_graph = nx_graph

        # create list of edges on the right panel
        self._current_graph = nx_graph
        self.populate_edge_list(nx_graph)
        self.populate_node_pickers(nx_graph)

        # remember which algorithm was used
        self.last_algo = algo

        # compute and show cross algorithm differences
        try:
            self.compute_and_show_comparison(algo)
        except Exception as _e:
            # keep the UI responsive even if comparison fails
            self.ui.statusBar().showMessage("Comparison failed; see console.")

        # stores just learned graph in result cache
        try:
            self._result_cache[self._graph_key(algo)] = nx_graph
        except Exception:
            pass

        for other in ("PC", "GFCI", "FGES"):
            if other != algo:
                try:
                    self._get_or_run_graph(other)
                except Exception:
                    pass

        try:
            self._build_comparison_text()
        except Exception:
            pass

        # tells user the search is done
        self.ui.statusBar().showMessage(f"{algo} finished.")



    def _validate_df(self, df):
        # return a list problems with the dataframe
        issues = []
        try:
            if df is None or df.shape[0] == 0 or df.shape[1] == 0:
                issues.append("No data rows or no columns.")
                return issues
            cols = [str(c) for c in list(df.columns)]
            if any(c.strip() == "" for c in cols):
                issues.append("Some column names are blank. The first row must be a header of variable names.")
            if len(set(cols)) != len(cols):
                issues.append("Duplicate column names found. Column names must be unique.")
    


            def _numlike(s):
                s = s.strip()
                if s.startswith("-"):
                    s = s[1:]
                return s.replace(".", "", 1).isdigit()
            


            if cols and all(_numlike(c) for c in cols):
                issues.append("Column names look numeric. This suggests the CSV has no header row.")
        except Exception:
            issues.append("Unexpected problem while inspecting the CSV structure.")
        return issues

    

    def draw_graph(self, nx_graph, algo_name, custom_title=""):
        # close old graphs
        if self._figure is not None:
            try: plt.close(self._figure)
            except Exception: pass


        # Qt canvas prep
        layout = self.ui.graphPage.layout() or QVBoxLayout(self.ui.graphPage)
        self.ui.graphPage.setLayout(layout)
        while layout.count():
            w = layout.takeAt(0).widget()
            if w:
                w.setParent(None)

        # figure and axes
        fig, ax = plt.subplots()

        # no tick marks or numbers on axes
        ax.set_xticks([]);  ax.set_yticks([])
        # dark frame around graph
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            
        # titles
        title_to_show = (custom_title or self.data_filename or "").strip()
        if title_to_show:
            fig.text(0.5, 0.97, title_to_show,
                    ha="center", va="top",
                    fontsize=14, weight="bold")

        # subtitles for each algorithm
        algo_up = str(algo_name).upper()
        if algo_up == "FGES":
            # show the FGES knob instead of alpha and depth
            subtitle = f"FGES  (penalty={getattr(self, 'penalty_discount', 2.0):g})"
        else:
            # PC / GFCI show alpha and depth
            subtitle = f"{algo_name}  (α={self.alpha}, depth={self.depth})"

        fig.text(0.5, 0.93,            
                subtitle,
                ha="center", va="top",
                fontsize=11)

        # push the plotting area down so the DAG starts lower leaving 15% top margin
        fig.subplots_adjust(top=0.85)


        # layout
        if set(nx_graph.nodes()) != set(self._pos):
            # brand new graph or node set changed makes fresh layout


            raw_pos = (self._graphviz_pos(nx_graph)
                    or nx.spring_layout(nx_graph, seed=42))

            # so nodes don't touch the frame, add 5% margin
            m      = 0.05
            xs, ys = zip(*raw_pos.values())
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            rngx = max(1e-9, maxx - minx)
            rngy = max(1e-9, maxy - miny)

            pos = {k: (m + (x - minx) / rngx * (1 - 2*m),
                    m + (y - miny) / rngy * (1 - 2*m))
                for k, (x, y) in raw_pos.items()}

            # store for dragging
            self._pos = {n: (float(x), float(y)) for n, (x, y) in pos.items()}

        # on redraw, reuse existing positions
        pos = self._pos

        # draw nodes as boxes with rounded corners
        self._node_artists.clear()
        for node, (x, y) in pos.items():
            artist = ax.text(
                x, y, str(node),
                ha="center", va="center", fontsize=10, picker=6,
                bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.15",
                        fc="#EAFBFF", ec="#333333", lw=1)
            )
            self._node_artists[node] = artist

        # first draw to measure the boxes
        fig.canvas.draw()
        renderer  = fig.canvas.get_renderer()
        shrink_pt = {}
        pt_per_px = 72.0 / fig.dpi
        for node, art in self._node_artists.items():
            bb      = art.get_window_extent(renderer)
            w_pt, h_pt = bb.width * pt_per_px, bb.height * pt_per_px
            shrink_pt[node] = 0.55 * 0.5 * math.hypot(w_pt, h_pt)

        # redraw edges
        def render_edges():
            for a in self._edge_artists:
                a.remove()
            self._edge_artists.clear()

            # helper: point a little in from an endpoint along the edge
            def _offset_point(x_from, y_from, x_to, y_to, dist):
                dx, dy = (x_from - x_to), (y_from - y_to)
                L = math.hypot(dx, dy) or 1e-9
                return (x_from - dx / L * dist, y_from - dy / L * dist)

            # choose a circle radius and offset in data coords
            # use ~2% of the axis span as a decent default
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            span = max(x1 - x0, y1 - y0)
            circ_r   = 0.02 * span
            circ_off = 0.03 * span

            for u, v in nx_graph.edges():
                (x1, y1), (x2, y2) = pos[u], pos[v]
                attrs = nx_graph.get_edge_data(u, v, default={})
                ep1 = attrs.get("ep1")
                ep2 = attrs.get("ep2")
                mark = attrs.get("mark")

                # base line trimmed so it does not overlap node boxes
                base = ax.annotate(
                    "", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle='-',
                        lw=1.2,
                        color="#333333",
                        shrinkA=shrink_pt[u] + 12,
                        shrinkB=shrink_pt[v] + 12,
                        connectionstyle="arc3,rad=0.0",
                    ),
                    zorder=3,
                )
                self._edge_artists.append(base)

                # arrowhead endpoints
                if ep2 == "ARROW":
                    arr_uv = ax.annotate(
                        "", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=1.2,
                            color="#333333",
                            mutation_scale=18,
                            shrinkA=shrink_pt[u] + 12,
                            shrinkB=shrink_pt[v] + 12,
                            connectionstyle="arc3,rad=0.0",
                        ),
                        zorder=4,
                    )
                    self._edge_artists.append(arr_uv)

                if ep1 == "ARROW":
                    arr_vu = ax.annotate(
                        "", xy=(x1, y1), xytext=(x2, y2),
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=1.2,
                            color="#333333",
                            mutation_scale=18,
                            shrinkA=shrink_pt[v] + 12,
                            shrinkB=shrink_pt[u] + 12,
                            connectionstyle="arc3,rad=0.0",
                        ),
                        zorder=4,
                    )
                    self._edge_artists.append(arr_vu)

                # circles placed near endpoints
                if ep2 == "CIRCLE":
                    cx, cy = _offset_point(x2, y2, x1, y1, circ_off)
                    circ = mpatches.Circle((cx, cy), radius=circ_r * 0.6,
                                        fill=False, ec="#333333", lw=1.2, zorder=5)
                    ax.add_patch(circ)
                    self._edge_artists.append(circ)

                if ep1 == "CIRCLE":
                    cx, cy = _offset_point(x1, y1, x2, y2, circ_off)
                    circ = mpatches.Circle((cx, cy), radius=circ_r * 0.6,
                                        fill=False, ec="#333333", lw=1.2, zorder=5)
                    ax.add_patch(circ)
                    self._edge_artists.append(circ)



        render_edges()

        # explicit limits so graph can't vanish
        xs, ys = zip(*pos.values())
        margin = 0.1
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

        # add figure to Qt
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        canvas.draw()

        self._figure = fig
        self._canvas = canvas



        # add drag handlers
        def node_under(event):
            if event.inaxes != ax:
                return None
            for name, art in self._node_artists.items():
                hit, _ = art.contains(event)
                if hit:
                    return name
            return None
        


        def set_cursor(cursor_type):
            canvas.setCursor(QCursor(cursor_type))



        def on_press(event):
            n = node_under(event)
            if n:
                self._dragging_node = n
                canvas.widgetlock(self)
                set_cursor(Qt.ClosedHandCursor)



        def on_release(event):
            if self._dragging_node:
                self._dragging_node = None
                canvas.widgetlock(None)
                # choose what cursor is showing
                set_cursor(Qt.OpenHandCursor if node_under(event) else Qt.ArrowCursor)



        def on_motion(event):
            # while dragging
            if self._dragging_node:
                if event.inaxes == ax and event.xdata is not None:
                    pos[self._dragging_node] = (event.xdata, event.ydata)
                    self._node_artists[self._dragging_node].set_position(pos[self._dragging_node])
                    render_edges()
                    canvas.draw_idle()
                # don't run hover logic
                return

            # hover feedback when not dragging
            set_cursor(Qt.OpenHandCursor if node_under(event) else Qt.ArrowCursor)



        # avoid duplicate connections
        cid_tag = "_drag_cids"
        if not hasattr(canvas, cid_tag):
            canvas.mpl_connect("button_press_event",   on_press)
            canvas.mpl_connect("button_release_event", on_release)
            canvas.mpl_connect("motion_notify_event",  on_motion)
            setattr(canvas, cid_tag, True)



    def _graphviz_pos(self, G: nx.DiGraph):
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
        except Exception as e:
            log.debug("GRAPHVIZ: nx_pydot not importable (%r) -> fallback", e)
            return None

        # first try original labels
        try:
            pos = graphviz_layout(G, prog="dot")
            if pos:
                log.debug("GRAPHVIZ: used dot for %d nodes", len(pos))
                return pos
            log.debug("GRAPHVIZ: returned empty -> fallback")
        except Exception as e:
            log.debug("GRAPHVIZ: dot failed (%r) -> retry with safe IDs", e)
            print("graphviz_layout failed; retrying with safe IDs:", repr(e))

        # retry with the safe IDs
        mapping = {n: f"n{idx}" for idx, n in enumerate(G.nodes())}
        H = nx.relabel_nodes(G, mapping, copy=True)
        try:
            pos_safe = graphviz_layout(H, prog="dot")
            # if still empty
            if not pos_safe:
                return None
            inv = {v: k for k, v in mapping.items()}
            log.debug("GRAPHVIZ: safe-ID dot OK for %d nodes", len(pos_safe))
            return {inv[k]: v for k, v in pos_safe.items()}
        except Exception as e2:
            print("graphviz_layout still failed:", repr(e))
            log.debug("GRAPHVIZ: safe-ID dot failed (%r) -> fallback", e2)
            return None
        


    # fill right hand list with directed edges u --> v using nx_graph
    def populate_edge_list(self, nx_graph):
        lst = getattr(self.ui, "edgeList", None)
        if lst is None:
            # if UI does not have the list
            return

        lst.clear()

        if nx_graph is None or nx_graph.number_of_edges() == 0:
            lst.addItem("(no directed edges)")
            return

        # build readable list like: A --> B
        def _edge_label(u, v):
            attrs = nx_graph.get_edge_data(u, v, default={})
            mark = attrs.get("mark")
            return f"{u} {mark} {v}" if mark else f"{u} --> {v}"
        
        items = [_edge_label(u, v) for (u, v) in nx_graph.edges()]
        lst.addItems(items)



    # populate two combo boxes on the "Why not connected?" tab on the right
    def populate_node_pickers(self, nx_graph):
        A = getattr(self.ui, "explainA", None)
        B = getattr(self.ui, "explainB", None)
        if A is None or B is None:
            return
        
        names = sorted(str(n) for n in nx_graph.nodes())

        A.blockSignals(True); B.blockSignals(True)
        A.clear(); B.clear()
        A.addItems(names); B.addItems(names)
        A.blockSignals(False); B.blockSignals(False)



    # show edge explanation for two chosen variables
    def show_edge_explanation(self, item):
        text = item.text().strip()

        parsed = _parse_edge_label(text)
        if not parsed:
            QtWidgets.QMessageBox.information(self.ui, "Edge", text)
            return

        A, mark, B = parsed
        tpl = _EDGE_TEMPLATES.get(mark)
        if not tpl:
            QtWidgets.QMessageBox.information(self.ui, "Edge", text)
            return

        html_msg = _format_edge_explanation(tpl, A, B)
        EdgeExplanationDialog.show(self.ui, html_msg, title="Edge explanation")



    # explain why A and B aren't connected and show a separating set
    def explain_non_edge(self):
        # pull UI widgets
        A = self.ui.explainA.currentText().strip()
        B = self.ui.explainB.currentText().strip()
        T = self.ui.explainText

        if not A or not B or A == B:
            T.setPlainText("Pick two different variables.")
            return

        G = getattr(self, "_current_graph", None)
        if G is None:
            T.setPlainText("Run a search first.")
            return

        # if graph already has an edge, say so
        if G.has_edge(A, B):
            T.setPlainText(f"'{A} → {B}' is present in the learned graph.")
            return
        if G.has_edge(B, A):
            T.setPlainText(f"'{B} → {A}' is present in the learned graph.")
            return

        # candidate pool is neighbors of A or B in the graph
        cand = sorted(
            set(G.predecessors(A)) | set(G.successors(A)) |
            set(G.predecessors(B)) | set(G.successors(B))
        )
        # fallback to other variables if that is too small
        if len(cand) < 2:
            cand = [n for n in G.nodes() if n not in (A, B)]

        # respect depth but cap at three for efficiency
        max_k = 3 if self.depth == -1 else max(0, min(self.depth, 3))

        # on demand Python sepset finding
        found = None
        try:
            found = self.tetrad.find_sepset_py(A, B, candidates=cand, max_k=max_k)
        except Exception:
            found = None

        # choose the label for the test actually used
        dtype = self.dtype
        if dtype == "auto":
            dtype = self.tetrad._effective_dtype(self.tetrad._last_df)

        test_name = "Fisher-Z (partial corr)" if dtype == "cont" else "Conditional χ² (G²)"
        header = f"Test: {test_name}  (α = {self.alpha}, depth = {self.depth})"

        # prints sepset if found for explanation, otherwise explains likely reasons for no edge
        if found:
            S, pval = found
            ptxt = (f"p = {pval:.4g}" if pval is not None else "p-value not available")
            T.setPlainText(
                f"No edge between {A} and {B}.\n"
                f"One separating set is S = {{{', '.join(S)}}} ({ptxt}).\n"
                f"{header}\n\n"
                "Interpretation: conditioned on S, the test did not find evidence of dependence."
            )
        else:
            T.setPlainText(
                "No direct edge was learned between these variables.\n"
                "A separating set may exist but wasn’t found among small candidate sets, "
                "or the score stage penalized the link.\n"
                f"{header}"
            )



    # generates text for compare tab and displays it
    def compute_and_show_comparison(self, base_algo: str):
        T = getattr(self.ui, "compareText", None)
        if T is None:
            return
        try:
            T.setPlainText(self._build_comparison_text())
        except Exception:
            T.setPlainText("Comparison failed; check console for details.")



    # produces blueprint of the current dataset for caching
    def _df_signature(self):
        if self.df is None:
            return ("<no-data>", 0, 0, ())
        try:
            cols = tuple(map(str, list(self.df.columns)))
        except Exception:
            cols = ()
        return (
            getattr(self, "data_filename", None) or "<memory>",
            len(self.df),
            len(cols),
            cols,
        )



    # current parameters put into a tuple
    def _param_signature(self):
        return (
            float(getattr(self, "alpha", 0.05)),
            int(getattr(self, "depth", -1)),
            str(getattr(self, "dtype", "auto")),
            float(getattr(self, "penalty_discount", 2.0)),  # FGES knob
        )



    def _graph_key(self, algo: str):
        return (algo.upper(), self._df_signature(), self._param_signature())
    


    def _compare_key(self):
        return (self._df_signature(), self._param_signature())
    


    # clears both caches when either data or parameters change
    def _invalidate_caches(self):
        self._result_cache.clear()
        self._compare_cache.clear()
        self._last_compare_key = None
        log.debug("CACHE: invalidated  df=%s  params=%s", self._df_signature(), self._param_signature())



    # help button causing help instructions pop up box
    def show_help(self):
        QMessageBox.information(self.ui, "Help", "Select a CSV then choose an algorithm.")



    # export current graph to a png file
    def export_graph_png(self):
        if self._figure is None or self._canvas is None:
            QMessageBox.information(self.ui, "No graph", "Run a search first, then export.")
            return

        # choose filename
        default_base = (self.graph_title or getattr(self, "data_filename", "") or "graph").strip()
        default_base = default_base.rsplit(".", 1)[0] or "graph"
        path, _ = QFileDialog.getSaveFileName(
            self.ui, "Export DAG to PNG", f"{default_base}.png",
            "PNG Image (*.png)"
        )
        if not path:
            return

        # render latest drags
        self._canvas.draw()

        t0 = time.perf_counter()
        try:
            # save what you see with tiny padding to avoid clipping black box
            self._figure.savefig(
                path,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.02,
                facecolor="white",
                edgecolor="none",
            )
            dt_ms = (time.perf_counter() - t0) * 1000
            print(f"PERF export_png_ms={dt_ms:.1f}  file={path}")
            self.ui.statusBar().showMessage(f"Saved PNG: {path}")
        except Exception as e:
            QMessageBox.critical(self.ui, "Export failed", str(e))
        


    
    # export graphs edges and attributes to a txt file
    def export_edges_txt(self):
        """Save the current graph (nodes, edges, parameters) to a .txt file."""
        if getattr(self, "_current_graph", None) is None:
            QtWidgets.QMessageBox.information(self.ui, "No graph", "Run a search first, then export.")
            return

        G = self._current_graph

        # default filename base from graph title or CSV name
        default_base = (getattr(self, "graph_title", "") or getattr(self, "data_filename", "") or "graph").strip()
        default_base = os.path.splitext(default_base)[0] or "graph"

        path, _ = QFileDialog.getSaveFileName(
            self.ui,
            "Export edges to TXT",
            f"{default_base}_edges.txt",
            "Text files (*.txt)"
        )
        if not path:
            return

        # build section, nodes alphabetical
        nodes = sorted(map(str, G.nodes()))
        nodes_line = ";".join(nodes) if nodes else "(none)"

        # edges with endpoint marks (e.g. -->, o->, o-o)
        def _edge_mark(u, v):
            data = G.get_edge_data(u, v, default={})
            return data.get("mark", "-->")

        edges = sorted([(str(u), str(v)) for (u, v) in G.edges()],
                    key=lambda p: (p[0].lower(), p[1].lower()))
        edge_lines = [f"{i}. {u} {_edge_mark(u, v)} {v}" for i, (u, v) in enumerate(edges, start=1)]
        edges_block = "\n".join(edge_lines) if edge_lines else "(no edges)"

        # graph attributes / run parameters
        params = []
        if hasattr(self, "last_algo"):
            params.append(f"Algorithm: {self.last_algo}")

        # always show data type
        if hasattr(self, "dtype"):
            params.append(f"data_type: {self.dtype}")

        # PC/GFCI: show alpha/depth, FGES: show penalty
        algo = getattr(self, "last_algo", "")
        if algo in ("PC", "GFCI"):
            if hasattr(self, "alpha"):
                params.append(f"alpha: {self.alpha}")
            if hasattr(self, "depth"):
                params.append(f"depth: {self.depth}")
        if algo == "FGES":
            params.append(f"penalty_discount: {getattr(self, 'penalty_discount', 2.0)}")

        if hasattr(self.tetrad, "last_bic") and self.tetrad.last_bic is not None:
            params.append(f"BIC: {self.tetrad.last_bic}")

        param_block = "\n".join(params) if params else "(none)"

        # assemble file text
        text = (
            "Graph Nodes:\n"
            f"{nodes_line}\n\n"
            "Graph Edges:\n"
            f"{edges_block}\n\n"
            "Graph Attributes:\n"
            f"{param_block}\n"
        )

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.ui.statusBar().showMessage(f"Saved edges: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.ui, "Export failed", str(e))



    # generate comparison report text
    def _build_comparison_text(self) -> str:
        if self.df is None:
            return "No data loaded."

        compare_key = self._compare_key()
        if compare_key in self._compare_cache:
            self._last_compare_key = compare_key
            return self._compare_cache[compare_key]

        # get cached graphs or run algorithms once
        algos = ["PC", "GFCI", "FGES"]
        results = {}
        for a in algos:
            try:
                results[a] = self._get_or_run_graph(a)
            except Exception as e:
                results[a] = e

        # build edge sets with endpoint marks
        def _edges_with_marks(G):
            if not hasattr(G, "edges"):
                return set()
            s = set()
            for u, v, data in G.edges(data=True):
                mark = data.get("mark", "-->")
                s.add((str(u), mark, str(v)))
            return s

        # maps each algorithm to its set of edges
        E = {a: _edges_with_marks(g) for a, g in results.items() if hasattr(g, "edges")}

        # the parameters the comparison was ran under
        header = [f"Compared at α={self.alpha}, depth={self.depth}, dtype={self.dtype}"]
        if "FGES" in E:
            header.append(f"(FGES penalty={getattr(self, 'penalty_discount', 2.0)})")
        header = " ".join(header)

        counts = []
        for a in algos:
            if a in E:
                counts.append(f"{a}: {len(E[a])} edges")
            else:
                err = results.get(a)
                counts.append(f"{a}: ERROR ({err})")
        counts_block = " | ".join(counts)

        def fmt(t): return f"{t[0]} {t[1]} {t[2]}"
        sections = []
        for a in algos:
            if a not in E:
                continue
            others = set().union(*[E[b] for b in E if b != a]) if len(E) > 1 else set()
            # unique edges only one algorithm has
            uniq = sorted((fmt(t) for t in (E[a] - others)), key=str.lower)
            if uniq:
                sections.append(f"{a} only:\n  " + "\n  ".join(uniq))

        # orientation/type disagreements
        pairs = {}
        for a, es in E.items():
            for (u, mark, v) in es:
                key = tuple(sorted([u, v]))
                pairs.setdefault(key, {})
                pairs[key][a] = (u, mark, v)

        disagreements_blocks = []
        order = ["PC", "GFCI", "FGES"]
        for (x, y), per in sorted(pairs.items()):
            if len(per) < 2:
                continue
            # fetches only the endpoint mark of each algorithm
            marks = {a: per[a][1] for a in per}
            # if all algorithms used the same mark then there is no disagreement
            if len(set(marks.values())) <= 1:
                continue
            # otherwise build a disagreement block for the report
            lines = [f"{x} vs {y}:"]
            for a in order:
                if a in per:
                    u, mk, v = per[a]
                    lines.append(f"  {a}: {u} {mk} {v}")
            disagreements_blocks.append("\n".join(lines))

        # any disagreements found becomes a section in the report
        if disagreements_blocks:
            sections.append("Orientation/type disagreements:\n" + "\n\n".join(disagreements_blocks))

        # if no differences were found, print a "no differences" note
        if not sections:
            note = "No differences - all algorithms agree on edges and marks at current settings."
            if any(a in E for a in ("PC", "GFCI")):
                note += "\n(Heads-up: FGES is score-based and does not use α; PC/GFCI do.)"
            body = note
        else:
            body = "\n\n".join(sections)

        # metadata appended at the end
        meta = []
        if getattr(self, "graph_title", ""):
            meta.append(f"Title: {self.graph_title}")
        if getattr(self, "data_filename", ""):
            meta.append(f"Data: {self.data_filename}")
        footer = ("\n\n" + "\n".join(meta)) if meta else ""

        # creates the final report
        text = f"{header}\n{counts_block}\n\n{body}{footer}"

        # cache comparison text so it is not rebuilt repeatedly
        self._compare_cache[compare_key] = text
        self._last_compare_key = compare_key
        return text
    


    def _get_or_run_graph(self, algo: str):
        key = self._graph_key(algo)
        if key in self._result_cache:
            log.debug("CACHE HIT: %s  key=%s", algo, key)
            return self._result_cache[key]
        log.debug("CACHE MISS: %s  key=%s  -> run_search()", algo, key)
        # reuse existing runner (no fresh TetradRunner())
        g = self.tetrad.run_search(
            self.df, algo,
            alpha=self.alpha, depth=self.depth, dtype=self.dtype,
            penalty_discount=getattr(self, "penalty_discount", 2.0),
        )
        self._result_cache[key] = g
        return g

        

    def export_comparisons_txt(self):
        # export not allowed if no dataset is present
        if self.df is None:
            QtWidgets.QMessageBox.information(self.ui, "No data", "Load data and run a search first.")
            return

        # builds comparison text if not present already in cache
        key = self._compare_key()
        text = self._compare_cache.get(key)
        if text is None:
            # builds once and uses cached graphs if present
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
            self.ui.statusBar().showMessage("Building comparison…")
            try:
                text = self._build_comparison_text()
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
                self.ui.statusBar().clearMessage()

        # picks default filename
        default_base = (self.graph_title or getattr(self, "data_filename", "") or "graph").strip()
        default_base = os.path.splitext(default_base)[0] or "graph"

        # open save as dialog with the automatic filename
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.ui, "Export comparisons to TXT",
            f"{default_base}_compare.txt",
            "Text files (*.txt)"
        )
        if not path:
            return

        # writes report to disk
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            self.ui.statusBar().showMessage(f"Saved comparisons: {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.ui, "Export failed", str(e))




    # when user clicks the "Tutorials" button in the toolbar
    def show_tutorials(self):
        # create tutorial hub dialog
        dlg = TutorialsHub(self.ui)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        
        # which tutorial was selected
        sel = dlg.selected_title()

        if sel == "Import Data":
            ImportWizard(self.ui).exec_()
        elif sel == "PC Algorithm":
            PcWizard(self.ui).exec_()
        elif sel == "GFCI Algorithm":
            GfciWizard(self.ui).exec_()
        elif sel == "FGES Algorithm":
            FgesWizard(self.ui).exec_()
        elif sel == "Constraint vs Score Methods":
            try:
                from dialogs import CbVsSbWizard
                CbVsSbWizard(self.ui).exec_()
            except Exception:
                QtWidgets.QMessageBox.information(
                    self.ui, "Coming soon",
                    "This tutorial will be added soon."
                )