from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy, QToolButton, QStyle
import pathlib



# tutorial for importing CSV files
class ImportWizard(QtWidgets.QDialog):

    # HTML strings for each page of the tutorial
    # page 2 left empty for custom widget
    PAGES = [
        # page 1
        """<span style="color:#003366; font-size:20pt; font-weight:bold;">
Picking the right file</span><br>
<ul style="margin-left:-20px; margin-top:8px;">
  <li>File must end in <code>.csv</code> (comma separated values).</li>
  <li>One header row containing column names (A, B, C...).</li>
  <li>Each further row = one measurement / observation.</li>
</ul>""",

        # page 2
        "",

        # page 3
        """<span style="color:#003366; font-size:20pt; font-weight:bold;">
Missing values?</span><br>
Leave a blank cell or write <code>NA</code>. Rows containing NA will be skipped.<br><br>
When you're ready, click <b>Continue…</b> to choose your file."""
    ]



    def __init__(self, parent=None):
        super().__init__(parent)

        # set dialog title and minimum width
        self.setWindowTitle("Importing CSV Data")
        self.setMinimumWidth(600)

        self.stack = QtWidgets.QStackedWidget(self)

        # build page 1, custom page 2 with image, and page 3
        for i, txt in enumerate(self.PAGES):
            if i == 1:
                page = self._page_sample()  # ← image page
            else:
                lab = QtWidgets.QLabel(txt); lab.setWordWrap(True)
                page = QtWidgets.QWidget()
                QtWidgets.QVBoxLayout(page).addWidget(lab)
            self.stack.addWidget(page)

        # navigation buttons
        self.backBtn   = QtWidgets.QPushButton("Back")
        self.nextBtn   = QtWidgets.QPushButton("Next")
        self.cancelBtn = QtWidgets.QPushButton("Cancel")

        # connect clicking to helper methods
        self.backBtn.clicked.connect(self._prev)
        self.nextBtn.clicked.connect(self._next)
        self.cancelBtn.clicked.connect(self.reject)

        # arrange buttons and pages in the layout
        nav = QtWidgets.QHBoxLayout()
        nav.addWidget(self.backBtn)
        nav.addWidget(self.nextBtn)


        foot = QtWidgets.QHBoxLayout()
        # adds cancel button to the left
        foot.addWidget(self.cancelBtn)
        foot.addStretch()
        foot.addLayout(nav)

        main = QtWidgets.QVBoxLayout(self)
        main.addWidget(self.stack)
        main.addLayout(foot)

        # set initial button styles
        self._refresh()



    # sample csv on page 2
    def _page_sample(self) -> QtWidgets.QWidget:
        # create blank widget
        w  = QtWidgets.QWidget()
        # give it vertical layout
        v  = QtWidgets.QVBoxLayout(w)
        # align the contents at the top
        v.setAlignment(Qt.AlignTop)

        # navy blue heading
        heading = QtWidgets.QLabel(
            '<span style="color:#003366; font-size:20pt; font-weight:bold;">'
            'What a CSV looks like</span>'
        )
        heading.setWordWrap(True)
        v.addWidget(heading)

        # label shows the PNG file
        img_label = QtWidgets.QLabel()
        img_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        # the path where sample_csv.png is
        img_path = pathlib.Path(__file__).resolve().parent / "sample_csv.png"

        # if file exists then load it, if not then placeholder text is shown
        if img_path.exists():
            pix = QtGui.QPixmap(str(img_path))
            # scale to a certain width
            pix = pix.scaledToWidth(440, Qt.SmoothTransformation)
            img_label.setPixmap(pix)
        else:
            img_label.setText("<i>(Place <code>sample_csv.png</code> next to dialogs.py)</i>")

        v.addWidget(img_label)
        v.addStretch(1)
        return w
    


    # nav helpers
    # go previous page if not on first already
    def _prev(self):
        i = self.stack.currentIndex()
        if i > 0:
            self.stack.setCurrentIndex(i - 1)
        self._refresh()



    # go to the next page or close if on last page
    def _next(self):
        last = (self.stack.currentIndex() == self.stack.count() - 1)
        if last:
            self.accept()
        else:
            self.stack.setCurrentIndex(self.stack.currentIndex() + 1)
            self._refresh()

            

    # finds the page index and if it is the last
    def _refresh(self):
        i    = self.stack.currentIndex()
        last = (i == self.stack.count() - 1)
        # get rid of the back button on the first page
        self.backBtn.setEnabled(i > 0)
        self.nextBtn.setText("Continue…" if last else "Next")
        self.nextBtn.setMinimumWidth(100)
        # green continue button
        self.nextBtn.setStyleSheet(
            "QPushButton { background:#4CAF50; color:white; "
            "border:1px solid #3c8f45; border-radius:6px; padding:6px 14px; font-weight:bold; }"
            "QPushButton:hover { background:#66BB6A; }"
        ) if last else self.nextBtn.setStyleSheet("")



# tutorial for the PC algorithm
class PcWizard(QtWidgets.QDialog):

    # HTML strings for each page of the tutorial
    PAGES = [
        # page 1
        """<span style="color:#003366; font-size:20pt; font-weight:bold;">
What is the PC algorithm?</span><br>
A two-stage procedure that learns a causal graph by using observational data<br><br>
<b>1.&nbsp;&nbsp;Pruning (skeleton search)</b><br>
The PC algorithm starts with every variable fully connected to each other and any edges that contradict a conditional independence found in the data are removed systematically. This step leaves only statistically tested relationships.<br><br>
<b>2.&nbsp;&nbsp;Orientation (arrow assignment)</b><br>
The algorithm now applies a set of logical orientation rules (Meek rules and collider detection) to determine arrow direction when the data allows. Edges with no determined direction are left undirected.<br><br>
This outputs a Directed Acyclic Graph (DAG) that represents one of the possible causal models that is consistent with the observational data.""",

        # page 2
        """<span style="color:#003366; font-size:20pt; font-weight:bold;">
Assumptions of the PC algorithm</span><br>
<b>1.&nbsp;&nbsp;Causal Markov</b><br>
Every variable is independent of its non-effects after you discover its direct causes. Allows the PC algorithm to determine causal effects from conditional independence tests. If untrue for your system, a different model class is required.<br><br>
<b>2.&nbsp;&nbsp;Faithfulness</b><br>
Every statistical independency in your dataset match exactly the d-separations in the true DAG so no paths are mistakenly cancelled. This ensures discovered independence maps to the removal of an edge. If this assumption is violated, real links can be lost, or arrows can be wrongly orientated.<br><br>
<b>3.&nbsp;&nbsp;Causal sufficiency</b><br>
There is no hidden common cause of any variables included in your data. PC assumes the graph has no latent confounders. If you expect hidden variables, FCI or RFCI are better algorithms to use that account for this.<br><br>
<b>4.&nbsp;&nbsp;Independent, identically distributed samples</b><br>
Observations are not drawn from a feedback time series or a non-stationary process. for time series data, extensions like PCMCI are more appropriate to use.""",

        # page 3
        """<span style="color:#003366; font-size:20pt; font-weight:bold;">
Strengths</span><br>
<b>Transparent reasoning:</b> Every edge deletion is due to a concrete conditional independence test, meaning each step can be inspected and replicated<br><br>
<b>Scales well to hundreds of variables</b><br><br>
<b>Extensible framework:</b> FCI / RFCI and PCMCI and many modern hybrids (e.g. GFCI) use the same two stage idea<br><br>
<span style="color:#003366; font-size:20pt; font-weight:bold;">
Weaknesses</span><br>
<b>Sensitive to sample size and faithfulness:</b> small tweaks in α or a cancelled path can flip edge directions or drop associations<br><br>
<b>Only recovers a Markov equivalence class:</b> some arrows can stay undirected and require extra experiments or background information to be fully orientated<br><br>
<b>Assumes causal sufficiency:</b> violating this can produce false edges. If you suspect latent cofounders, use FCI-style algorithms"""
    ]



    def __init__(self, parent=None):
        super().__init__(parent)

        # window title and minimum width
        self.setWindowTitle("PC Search tutorial")
        self.setMinimumWidth(480)

        # shows one page at a time
        self.stack = QtWidgets.QStackedWidget(self)

        # loop through HTML strings for each page
        for txt in self.PAGES:
            lab = QtWidgets.QLabel(txt)
            lab.setWordWrap(True)
            page = QtWidgets.QWidget()
            QtWidgets.QVBoxLayout(page).addWidget(lab)
            self.stack.addWidget(page)

        # for widgets for navigation
        self.backBtn   = QtWidgets.QPushButton("Back")
        self.nextBtn   = QtWidgets.QPushButton("Next")
        self.cancelBtn = QtWidgets.QPushButton("Cancel")
        self.dontShow  = QtWidgets.QCheckBox("Don’t show this again")

        # connect clicking functionality
        self.backBtn.clicked.connect(self._prev)
        self.nextBtn.clicked.connect(self._next)
        self.cancelBtn.clicked.connect(self.reject)

        # button layout
        nav = QtWidgets.QHBoxLayout()
        nav.addWidget(self.backBtn)
        nav.addWidget(self.nextBtn)

        foot = QtWidgets.QHBoxLayout()
        # cancel shown far left of the window
        foot.addWidget(self.cancelBtn)
        foot.addStretch()
        foot.addWidget(self.dontShow)
        foot.addLayout(nav)

        # vertical layout
        main = QtWidgets.QVBoxLayout(self)
        main.addWidget(self.stack)
        main.addLayout(foot)

        # set initial button styles
        self._refresh()



    # helpers
    # previous page unless on first page
    def _prev(self):
        i = self.stack.currentIndex()
        if i > 0:
            self.stack.setCurrentIndex(i - 1)
        self._refresh()



    # next page unless on last page then continue
    def _next(self):
        last = self.stack.currentIndex() == self.stack.count() - 1
        if last:
            self.accept()
        else:
            self.stack.setCurrentIndex(self.stack.currentIndex() + 1)
            self._refresh()


    # refresh buttons to change back and next if on either first or last page
    def _refresh(self):
        i    = self.stack.currentIndex()
        last = (i == self.stack.count() - 1)
        self.backBtn.setEnabled(i > 0)
        self.nextBtn.setText("Continue…" if last else "Next")
        self.nextBtn.setMinimumWidth(100)
        # paint the continue button green
        self.nextBtn.setStyleSheet(
            "background:#4CAF50; color:white; border-radius:6px; font-weight:bold;"
            if last else ""
        )


    # user clicked don't show again
    def dontShowAgain(self) -> bool:
        return self.dontShow.isChecked()



# tutorial for the GFCI algorithm
class GfciWizard(QtWidgets.QDialog):
    """Three-page GFCI tutorial (wording fixed by spec)."""
    # HTML strings for each page of the tutorial
    PAGES = [
        # page 1
        """<span style="color:#003366; font-size:20pt; font-weight:bold;">
What is the GFCI Algorithm?</span><br>
A hybrid causal discovery algorithm which combines score-based search with constraint-based tests on observational data to learn a causal graph<br><br>
<b>1.&nbsp;&nbsp;Greedy search (proposal stage)</b><br>
GFCI begins by using score-based tests (e.g. BIC/SEM-BIC or BDeu) to find a high scoring DAG structure. This stage suggests which variables likely relate to one another and strikes a balance between model fit and simplicity.<br><br>
<b>2.&nbsp;&nbsp;Independence tests and orientation (validation stage)</b><br>
The second stage uses conditional independence testing to remove edges which the data contradicts. Logical orientation rules then assign directions to the arrows which remain.<br><br>
A causal graph is produced which is consistent with your data. GFCI can represent the possibility of hidden confounders being present by using partially orientated edges.""",

        # page 2
        """<span style="color:#003366; font-size:20pt; font-weight:bold;">
Assumptions of the GFCI algorithm</span><br>
<b>1.&nbsp;&nbsp;Causal Markov</b><br>
Every variable is independent of its non-effects after you discover its direct causes. This lets the GFCI algorithm determine causal effects from conditional independence tests. If untrue for your system, a different model class is required as the statistical tests will not cleanly map to edges.<br><br>
<b>2.&nbsp;&nbsp;Faithfulness</b><br>
Statistical independencies in your dataset match exactly to the d-separations in the true causal graph so no paths are mistakenly cancelled. This ensures discovered independencies map to the removal of an edge. If this assumption is violated, real links can be lost, or arrows can be wrongly orientated.<br><br>
<b>3.&nbsp;&nbsp;Latent confounders allowed</b><br>
Unlike PC, GFCI can work with hidden common causes. Partially oriented edges are used to represent this uncertainty. Although, it assumes no selection bias meaning your sample has not been filtered in a way that causes false dependencies. If strong selection effects are expected, then results should be analysed with caution.<br><br>
<b>4.&nbsp;&nbsp;Independent, identically distributed samples</b><br>
Observations are not drawn from a feedback time series or a non-stationary process. for time series data, extensions like PCMCI are more appropriate to use.<br><br>
<b>5.&nbsp;&nbsp;Score-model adequacy</b><br>
GFCI combines a score step with statistical tests, so it is assumed your scoring model fits your data. E.g. SEM-BIC suits linear/continuous data with gaussian noise and BDeu suits discrete variables. The greedy score stage may produce the wrong structure if your scoring model does not match your data, even with appropriate independence tests.<br><br>
Choose tests/scores that match your data type (e.g. Fisher-Z/SEM-BIC for roughly linear continuous data, X2/BDeu for discrete data). Wrong edges can appear with mismatched choices.""",

        # page 3
        """<span style="color:#003366; font-size:20pt; font-weight:bold;">
Strengths</span><br>
<b>Handles hidden confounders:</b> uncertainty caused by unmeasured common causes is represented with partially oriented edges instead of forcing a wrongly directed causal relationship<br><br>
<b>Hybrid decisions:</b> global score and conditional independence tests are combined giving a balanced, stable graph<br><br>
<b>Transparent reasoning:</b> independence tests and scores support edges, so you can inspect why links are added or removed<br><br>
<b>Works across different data types</b><br><br>
<b>Appropriate for medium to large problems</b><br><br>
<span style="color:#003366; font-size:20pt; font-weight:bold;">
Weaknesses</span><br>
<b>Score-model dependencies:</b> if chosen score does not match your data, GFCI can fit the wrong structure<br><br>
<b>Ambiguous directions remain:</b> some edges can be partially oriented because latent confounding is allowed<br><br>
<b>Sensitive to sample size and faithfulness</b><br><br>
<b>Assumptions still apply</b><br><br>
<b>Can be heavier than PC:</b> the score stage and validations add extra compute time on high dimensional data"""
    ]



    def __init__(self, parent=None):
        super().__init__(parent)

        # window title and minimum width
        self.setWindowTitle("GFCI Search tutorial")
        self.setMinimumWidth(480)

        self.stack = QtWidgets.QStackedWidget(self)

        # loop through the HTML strings for each page
        for txt in self.PAGES:
            lab = QtWidgets.QLabel(txt)
            lab.setWordWrap(True)
            page = QtWidgets.QWidget()
            QtWidgets.QVBoxLayout(page).addWidget(lab)
            self.stack.addWidget(page)

        # create the buttons
        self.backBtn   = QtWidgets.QPushButton("Back")
        self.nextBtn   = QtWidgets.QPushButton("Next")
        self.cancelBtn = QtWidgets.QPushButton("Cancel")
        self.dontShow  = QtWidgets.QCheckBox("Don’t show this again")

        # connect clicking functionality
        self.backBtn.clicked.connect(self._prev)
        self.nextBtn.clicked.connect(self._next)
        self.cancelBtn.clicked.connect(self.reject)

        # layout
        nav = QtWidgets.QHBoxLayout()
        nav.addWidget(self.backBtn)
        nav.addWidget(self.nextBtn)

        foot = QtWidgets.QHBoxLayout()
        foot.addWidget(self.cancelBtn)
        foot.addStretch()
        foot.addWidget(self.dontShow)
        foot.addLayout(nav)

        main = QtWidgets.QVBoxLayout(self)
        main.addWidget(self.stack)
        main.addLayout(foot)

        # sets initial button styles
        self._refresh()



    # helpers
    # previous page unless on first page
    def _prev(self):
        i = self.stack.currentIndex()
        if i > 0:
            self.stack.setCurrentIndex(i - 1)
        self._refresh()



    # next page unless on final page then show continue
    def _next(self):
        last = self.stack.currentIndex() == self.stack.count() - 1
        if last:
            self.accept()
        else:
            self.stack.setCurrentIndex(self.stack.currentIndex() + 1)
            self._refresh()



    # refreshes buttons to determine if next or previous buttons should show depending on page number
    def _refresh(self):
        i    = self.stack.currentIndex()
        last = (i == self.stack.count() - 1)
        self.backBtn.setEnabled(i > 0)
        self.nextBtn.setText("Continue…" if last else "Next")
        self.nextBtn.setMinimumWidth(100)
        self.nextBtn.setStyleSheet(
            "background:#4CAF50; color:white; border-radius:6px; font-weight:bold;"
            if last else ""
        )



    # returns checkbox state for controller to store the users preference
    def dontShowAgain(self) -> bool:
        return self.dontShow.isChecked()
    


# tutorial for the FGES algorithm
class FgesWizard(QtWidgets.QDialog):
    # HTML strings for each page of the tutorial (your exact wording)
    PAGES = [
        # page 1
        """<span style="color:#003366; font-size:20pt; font-weight:bold;">
What is the FGES algorithm?</span><br>
FGES (Fast Greedy Equivalence Search) is a score-based method which uses a greedy algorithm to improve a score (e.g. BIC) and orientates edges that are identified. A CPDAG (partially directed graph) is outputted.<br><br>
<b>How it works:</b><br>
<b>1.&nbsp;&nbsp;Forward phase (add edges)</b><br>
Begins with an empty graph and iteratively adds an edge that gives the largest score improvement (e.g. lowering BIC the most), updating the structure as it runs.<br><br>
<b>2.&nbsp;&nbsp;Backward phase (remove edges)</b><br>
From the solution in the forward phase, edges are removed which improve the score the most to prune links left by the forward pass.<br><br>
<b>3.&nbsp;&nbsp;Orientation</b><br>
Logical orientation rules are applied (e.g. Meek rules) to add direction to edges without contradicting the equivalence class. This results in a CPDAG with some fixed arrows and others undirected due to observational data alone not being enough to determine the relationship.""",

        # page 2
"""<span style="color:#003366; font-size:20pt; font-weight:bold;">
Scoring &amp; Penalty</span><br>
<b>Continuous data:</b> usually SEM-BIC<br><br>
<b>Discrete data:</b> BDeu or discrete BIC variants<br><br>
<b>Penalty discount (c):</b> increases the complexity penalty in BIC<br>
&emsp;&bull; Higher c -&gt; sparser graphs (fewer edges)<br>
&emsp;&bull; Lower c -&gt; denser graphs (more edges)<br><br>
<b>FGES does not use α (significance):</b> α and depth are for constraint-based CI testing (PC and parts of GFCI)<br><br>
<span style="color:#003366; font-size:20pt; font-weight:bold;">What you get:</span><br>
A compact, well oriented graph balancing fit vs simplicity""",

        # page 3
        """<span style="color:#003366; font-size:20pt; font-weight:bold;">
Assumptions of the FGES algorithm</span><br>
<b>1.&nbsp;&nbsp;Causal Markov and Acyclicity</b><br>
Data are generated by a DAG where conditional independencies follow from d-separation in the true graph<br><br>
<b>2.&nbsp;&nbsp;Faithfulness / local consistency</b><br>
Observed independencies come from the structure rather than accidental parameter cancellations. With enough data, the scoring criteria prefers the correct equivalence class<br><br>
<b>3.&nbsp;&nbsp;score-model adequacy</b><br>
the score must match the data<br><br>
<b>4.&nbsp;&nbsp;Independent, identically distributed samples</b><br>
Observations are independent and identically distributed and strong selection mechanisms can distort the results<br><br>
<b>5.&nbsp;&nbsp;Causal sufficiency</b><br>
No unmeasured confounders among the included variables. If it is likely there are latent confounders, other algorithms are designed for that (e.g. GFCI)""",

        # page 4
        """<span style="color:#003366; font-size:20pt; font-weight:bold;">
Strengths</span><br>
<b>Fast and scalable:</b> handles many variables efficiently via greedy search<br><br>
<b>Stable global decisions:</b> balances fit and simplicity well through a single interpretable penalty knob<br><br>
<b>Clear output:</b> CPDAG captures what is and is not identifiable from observational data<br><br>
<b>Works across data types:</b> continuous (SEM-BIC) and discrete (BDeu/BIC)<br><br>
<b>Good baseline:</b> often a strong start before analysing more deeply<br><br>
<span style="color:#003366; font-size:20pt; font-weight:bold;">
Weaknesses</span><br>
<b>Score mismatch risk:</b> the wrong score assumptions can produce non-existent edges and orientations<br><br>
<b>No latent confounding:</b> could wrongly mistake confounding variables for causation<br><br>
<b>Ambiguity remains:</b> can leave edges undirected<br><br>
<b>Penalty sensitivity:</b> too low = spurious edges; too high = missed true edges""",

        # page 5
        """<span style="color:#003366; font-size:20pt; font-weight:bold;">
Practical Tips</span><br>
<b>Choosing penalty:</b> start around 2.0 and either increase if the graph looks implausibly dense or decrease if it looks too sparse<br><br>
<b>Model fit check:</b> if residual patterns or domain knowledge suggest nonlinearity in the data, be cautious interpreting SEM-BIC results<br><br>
<b>Cross check:</b> use the “Compare” tab to contrast FGES with PC/GFCI<br><br>
<b>Interpretation aid:</b> the “Why not connected?” panel runs post-hoc conditional independence tests to find separating sets to explain missing edges"""
    ]



    def __init__(self, parent=None):
        super().__init__(parent)

        # window title and minimum width
        self.setWindowTitle("FGES Search tutorial")
        self.setMinimumWidth(480)

        # loop through the HTML strings for each page
        self.stack = QtWidgets.QStackedWidget(self)
        for txt in self.PAGES:
            lab = QtWidgets.QLabel(txt); lab.setWordWrap(True)
            page = QtWidgets.QWidget()
            QtWidgets.QVBoxLayout(page).addWidget(lab)
            self.stack.addWidget(page)

        # create the buttons
        self.backBtn   = QtWidgets.QPushButton("Back")
        self.nextBtn   = QtWidgets.QPushButton("Next")
        self.cancelBtn = QtWidgets.QPushButton("Cancel")
        self.dontShow  = QtWidgets.QCheckBox("Don’t show this again")

        # connect clicking functionality
        self.backBtn.clicked.connect(self._prev)
        self.nextBtn.clicked.connect(self._next)
        self.cancelBtn.clicked.connect(self.reject)

        # layout
        nav = QtWidgets.QHBoxLayout()
        nav.addWidget(self.backBtn)
        nav.addWidget(self.nextBtn)

        foot = QtWidgets.QHBoxLayout()
        foot.addWidget(self.cancelBtn)
        foot.addStretch()
        foot.addWidget(self.dontShow)
        foot.addLayout(nav)

        main = QtWidgets.QVBoxLayout(self)
        main.addWidget(self.stack)
        main.addLayout(foot)

        # sets initial button styles
        self._refresh()



    # helpers
    # previous page unless on first page
    def _prev(self):
        i = self.stack.currentIndex()
        if i > 0:
            self.stack.setCurrentIndex(i - 1)
        self._refresh()



    # next page unless on final page then show Continue
    def _next(self):
        last = self.stack.currentIndex() == self.stack.count() - 1
        if last:
            self.accept()
        else:
            self.stack.setCurrentIndex(self.stack.currentIndex() + 1)
            self._refresh()



    # refreshes buttons to determine if next or previous buttons should show depending on page number
    def _refresh(self):
        i = self.stack.currentIndex()
        last = (i == self.stack.count() - 1)
        self.backBtn.setEnabled(i > 0)
        self.nextBtn.setText("Continue…" if last else "Next")
        self.nextBtn.setMinimumWidth(100)
        self.nextBtn.setStyleSheet(
            "background:#4CAF50; color:white; border-radius:6px; font-weight:bold;"
            if last else ""
        )



    # returns checkbox state for controller to store the users preference
    def dontShowAgain(self) -> bool:
        return self.dontShow.isChecked()



class GfciParamDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, alpha=0.05, depth=-1, dtype="auto", initial_title=''):
        super().__init__(parent)
        
        self.setWindowTitle("Run GFCI: parameters")

        # graph title
        self.titleEdit = QtWidgets.QLineEdit()
        self.titleEdit.setPlaceholderText("Optional — shown above the graph")
        self.titleEdit.setText(initial_title)

        # setting alpha (significance level 0.001 to 0.2)
        self.alphaBox = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0.001, maximum=0.2)
        self.alphaBox.setValue(alpha)

        # setting depth (-1 to 10)
        self.depthBox = QtWidgets.QSpinBox()
        self.depthBox.setRange(-1, 10)
        self.depthBox.setValue(depth)

        # setting data type (continuous or discrete)
        self.dtypeBox = QtWidgets.QComboBox()
        self.dtypeBox.addItems([
            "Auto detect (use data)",
            "Discrete (χ²/BDeu)",
            "Continuous (Fisher-Z/SEM-BIC)"
        ])
        self.dtypeBox.setCurrentIndex({"auto":0,"discrete":1,"cont":2}[dtype])

        # helper strings
        alpha_tip = ("α (significance): probability you accept a false edge.\n"
                    "Lower α = stricter tests, fewer but stronger links.")
        depth_tip = ("Depth: how far the search looks when testing conditional "
                    "independencies.  -1 = no limit.")
        dtype_tip = ("Choose the statistical test:\n"
                    "• Discrete → χ²/BDeu\n"
                    "• Continuous → Fisher-Z/SEM-BIC\n"
                    "• Auto lets the software decide column-by-column.")
        
        # build the form
        form = QtWidgets.QFormLayout()
        form.addRow("Graph title", self.titleEdit)  
        form.addRow("⍺ (significance)", self._row_with_help(self.alphaBox, alpha_tip))
        form.addRow("Depth (-1 = unlimited)", self._row_with_help(self.depthBox, depth_tip))
        form.addRow("Data type", self._row_with_help(self.dtypeBox, dtype_tip))

        # layout
        btnBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok,
            QtCore.Qt.Horizontal, self)
        btnBox.button(QtWidgets.QDialogButtonBox.Ok).setText("Run")
        btnBox.accepted.connect(self.accept)
        btnBox.rejected.connect(self.reject)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(form)
        v.addWidget(btnBox)



    def _row_with_help(self, editor: QtWidgets.QWidget, tip: str) -> QtWidgets.QWidget:
        # a widget that adds a clickable "?" button
        h = QtWidgets.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(editor)

        # help buttons for each parameter
        btn = QToolButton()
        btn.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        btn.setToolTip("Click for info")
        # change cursor when hovering over
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedSize(22, 22)
        btn.clicked.connect(lambda *_: QtWidgets.QMessageBox.information(
            self, "Parameter help", tip))

        h.addWidget(btn)
        w = QtWidgets.QWidget(); w.setLayout(h)
        return w



    # return parameter settings
    def values(self):
        dtype_map = {0:"auto", 1:"discrete", 2:"cont"}
        return (
            self.alphaBox.value(),
            self.depthBox.value(),
            dtype_map[self.dtypeBox.currentIndex()],
            self.titleEdit.text().strip()
        )
    


# FGES search algorithm parameters
class FgesParamDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, penalty=2.0, dtype="auto", initial_title=""):
        super().__init__(parent)
        self.setWindowTitle("Run FGES: parameters")

        # optional graph title field
        self.titleEdit = QtWidgets.QLineEdit()
        self.titleEdit.setPlaceholderText("Optional — shown above the graph")
        self.titleEdit.setText(initial_title)

        # penalty discount input
        self.penaltyBox = QtWidgets.QDoubleSpinBox(decimals=2, minimum=0.1, maximum=10.0)
        self.penaltyBox.setSingleStep(0.1)
        self.penaltyBox.setValue(penalty)

        # choice of test type drop down
        self.dtypeBox = QtWidgets.QComboBox()
        self.dtypeBox.addItems([
            "Auto detect (use data)",
            "Discrete (BDeu/BIC)",
            "Continuous (SEM-BIC)"
        ])
        self.dtypeBox.setCurrentIndex({"auto":0, "discrete":1, "cont":2}[dtype])

        # parameter tool tips
        penalty_tip = ("Penalty discount c for BIC: larger c → sparser graphs.\n"
                       "Typical defaults are around 2.0.")
        dtype_tip   = ("Choose score family:\n"
                       "• Discrete → BDeu or Discrete-BIC\n"
                       "• Continuous → SEM-BIC\n"
                       "• Auto lets the software choose from your data")

        # arrange options in labelled rows
        form = QtWidgets.QFormLayout()
        form.addRow("Graph title", self.titleEdit)
        form.addRow("Penalty discount (c)", self._row_with_help(self.penaltyBox, penalty_tip))
        form.addRow("Data type", self._row_with_help(self.dtypeBox, dtype_tip))

        # navigation buttons
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok,
            QtCore.Qt.Horizontal, self)
        btns.button(QtWidgets.QDialogButtonBox.Ok).setText("Run")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(form)
        v.addWidget(btns)



    # parameter help row
    def _row_with_help(self, editor: QtWidgets.QWidget, tip: str) -> QtWidgets.QWidget:
        h = QtWidgets.QHBoxLayout(); h.setContentsMargins(0,0,0,0); h.addWidget(editor)
        btn = QToolButton(); btn.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
        btn.setToolTip("Click for info"); btn.setCursor(Qt.PointingHandCursor); btn.setFixedSize(22,22)
        btn.clicked.connect(lambda *_: QtWidgets.QMessageBox.information(self, "Parameter help", tip))
        h.addWidget(btn); w = QtWidgets.QWidget(); w.setLayout(h); return w



    # converts parameters to a tuple for the algorithm caller
    def values(self):
        dtype_map = {0:"auto", 1:"discrete", 2:"cont"}
        return (
            self.penaltyBox.value(),
            dtype_map[self.dtypeBox.currentIndex()],
            self.titleEdit.text().strip()
        )

    

# list of available tutorials the user can choose from
class TutorialsHub(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tutorials")
        self.setMinimumWidth(420)
        self.setMinimumHeight(360)

        # title
        title = QtWidgets.QLabel(
            '<span style="color:#003366; font-size:20pt; font-weight:bold;">Tutorials</span>'
        )
        title.setAlignment(Qt.AlignLeft)

        # list of tutorials
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.addItems([
            "Import Data",
            "PC Algorithm",
            "GFCI Algorithm",
            "FGES Algorithm",
        ])
        self.list.itemDoubleClicked.connect(self.accept)  # double-click opens

        # add buttons
        openBtn = QtWidgets.QPushButton("Open")
        closeBtn = QtWidgets.QPushButton("Close")
        openBtn.clicked.connect(self.accept)
        closeBtn.clicked.connect(self.reject)

        # layout
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(closeBtn)
        btns.addWidget(openBtn)

        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(title)
        v.addWidget(self.list, 1)
        v.addLayout(btns)
        


    # return selected tutorial
    def selected_title(self) -> str:
        item = self.list.currentItem()
        return item.text() if item else ""



# edge explanation
class EdgeExplanationDialog(QtWidgets.QDialog):
    def __init__(self, html_text: str, parent=None, title="Edge explanation"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)

        # Body styled like wizard pages (rich text, scroll if needed)
        view = QtWidgets.QTextBrowser()
        view.setOpenExternalLinks(True)
        view.setFrameShape(QtWidgets.QFrame.NoFrame)
        view.setHtml(html_text)

        # Footer buttons: single "Continue…" like the wizards' last page
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        ok = btns.button(QtWidgets.QDialogButtonBox.Ok)
        ok.setText("Continue…")
        ok.setObjectName("edgeContinue")
        ok.setCursor(Qt.PointingHandCursor)
        btns.accepted.connect(self.accept)

        v = QtWidgets.QVBoxLayout(self)
        v.addWidget(view, 1)
        v.addWidget(btns)

    @staticmethod
    def show(parent, html_text: str, title="Edge explanation"):
        dlg = EdgeExplanationDialog(html_text, parent=parent, title=title)
        return dlg.exec_()