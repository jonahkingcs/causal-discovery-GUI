# helper that reads .ui files to build the widgets
from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSizePolicy, QDockWidget, QListWidget, QWidget, QLabel, QTabWidget, QComboBox, QPushButton, QTextEdit, QHBoxLayout
from PyQt5.QtCore import Qt
from pathlib import Path

# controller class that connects the ui to functionality
from controllers import Controller

import sys, time



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # reads mainwindow.ui and instantiates the buttons and docks into the window
        BASE = Path(__file__).resolve().parent
        UI_FILE = BASE.parent / "ui" / "mainwindow.ui"
        QSS_FILE = BASE / "styles.qss"
        uic.loadUi(str(UI_FILE), self)

        # load and apply qss style sheet
        qapp = QtWidgets.QApplication.instance()
        if qapp:
            with open(QSS_FILE, "r", encoding="utf-8") as f:
                qapp.setStyleSheet(f.read())

        # let list of tools fill and stretch
        if self.toolDock.layout() is None:
            lay = QVBoxLayout(self.toolDock)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.toolList)

        # tools list permanent (no closing or moving)
        self.dockWidget.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)

        # lock tool bar in place so no dragging options
        self.mainToolBar.setMovable(False)
        self.mainToolBar.setFloatable(False)
        self.mainToolBar.setAllowedAreas(Qt.TopToolBarArea)
        self.mainToolBar.setContextMenuPolicy(Qt.NoContextMenu)  

        # dock starts wide enough
        self.resizeDocks([self.dockWidget], [280], Qt.Horizontal)

        # minimum so widget can't be squished too thin
        self.dockWidget.setMinimumWidth(220)

        # let list expand to fill the dock
        self.toolList.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # wide enough for GFCI Search
        self.toolList.setMinimumWidth(240)

        # right side "Relationships" dock with two tabs
        relDock = QDockWidget("Relationships", self)
        relDock.setObjectName("relationsDock")
        relDock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        relDock.setAllowedAreas(Qt.RightDockWidgetArea)

        tabs = QTabWidget()

        # tab 1 "Edges"
        tab_edges = QWidget()
        ve = QVBoxLayout(tab_edges); ve.setContentsMargins(6, 6, 6, 6)
        title = QLabel("<b>Edges</b>")
        edgeList = QListWidget(); edgeList.setObjectName("edgeList")
        ve.addWidget(title); ve.addWidget(edgeList, 1)

        # tab 2 "Why not connected?"
        tab_why = QWidget()
        vw = QVBoxLayout(tab_why); vw.setContentsMargins(6, 6, 6, 6)

        # why not connected variable picker
        pickRow = QHBoxLayout()
        explainA = QComboBox(); explainB = QComboBox()
        explainBtn = QPushButton("Explain")
        pickRow.addWidget(explainA); pickRow.addWidget(explainB); pickRow.addWidget(explainBtn)

        # text box for explanation
        explainText = QTextEdit(); explainText.setReadOnly(True)

        # add variable picker and textbox to layout
        vw.addLayout(pickRow)
        vw.addWidget(explainText, 1)

        # tab 3 "Compare"
        tab_compare = QWidget()
        vc = QVBoxLayout(tab_compare); vc.setContentsMargins(6, 6, 6, 6)

        compareText = QTextEdit(); compareText.setReadOnly(True)
        compareText.setObjectName("compareText")
        compareText.setLineWrapMode(QTextEdit.NoWrap)
        vc.addWidget(compareText, 1)

        # adds tabs to the right hand side
        tabs.addTab(tab_edges, "Edges")
        tabs.addTab(tab_why, "Why not connected?")
        tabs.addTab(tab_compare, "Compare")

        # expose to controller
        self.compareText = compareText

        relDock.setWidget(tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, relDock)

        # expose to controller
        self.edgeList     = edgeList
        self.explainA     = explainA
        self.explainB     = explainB
        self.explainText  = explainText
        self.explainBtn   = explainBtn


        # hands the built ui to controller to handle the logic
        self.controller = Controller(self)

        # connects the actions to controller methods      
        self.helpAction.triggered.connect(self.controller.show_help)
        self.tutorialsAction.triggered.connect(self.controller.show_tutorials)
        self.exportGraphAction.triggered.connect(self.controller.export_graph_png)
        self.exportEdgesAction.triggered.connect(self.controller.export_edges_txt)
        self.exportComparisonsAction.triggered.connect(self.controller.export_comparisons_txt)
        

        # links GUI logic to figure out what tool user has clicked
        self.toolList.itemClicked.connect(self.controller.handle_tool_click)
        self.edgeList.itemClicked.connect(self.controller.show_edge_explanation)
        self.explainBtn.clicked.connect(self.controller.explain_non_edge)



if __name__ == '__main__':
    APP_T0 = time.perf_counter()

    # creates Qt application object
    app = QApplication([])

    # builds ui window
    win = MainWindow()

    # shows the window
    win.show()

    QtCore.QTimer.singleShot(
        0,
        lambda: print(f"PERF app_first_paint_ms={(time.perf_counter()-APP_T0)*1000:.1f}")
    )
    sys.exit(app.exec_())

    # starts Qt event loop
    app.exec_()