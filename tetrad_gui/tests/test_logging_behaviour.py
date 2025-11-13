import logging
import types
import networkx as nx
import pandas as pd

from controllers import Controller

class _DummyUI: pass

def _dummy_graph(tag):
    G = nx.DiGraph()
    # Make edges differ per tag so cache keys are meaningful in logs
    if tag == "PC":
        G.add_edge("A", "B", ep1="TAIL", ep2="ARROW", mark="-->")
    elif tag == "GFCI":
        G.add_edge("B", "C", ep1="TAIL", ep2="ARROW", mark="-->")
    else:
        G.add_edge("C", "D", ep1="TAIL", ep2="ARROW", mark="-->")
    return G

def test_cache_hit_and_invalidate(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="tetrad_gui")

    c = Controller(_DummyUI())
    c.df = pd.DataFrame({"A":[0,1,0], "B":[1,0,1], "C":[0,0,1]})
    c.data_filename = "unit_testing.csv"
    c.alpha = 0.05; c.depth = -1; c.dtype = "auto"; c.penalty_discount = 2.0

    # Stub out heavy Java search with a cheap fake
    def fake_run(df, algo, **kw):
        return _dummy_graph(algo)
    c.tetrad.run_search = fake_run

    # 1st call -> MISS, 2nd call -> HIT (same key)
    g1 = c._get_or_run_graph("PC")
    g2 = c._get_or_run_graph("PC")
    assert g1 is g2

    # Change a parameter -> invalidates keys, next call should MISS again
    c.alpha = 0.01
    c._invalidate_caches()
    g3 = c._get_or_run_graph("PC")

    txt = caplog.text
    assert "CACHE MISS: PC" in txt
    assert "CACHE HIT: PC"  in txt
    assert "CACHE: invalidated" in txt

def test_graphviz_fallback_logs(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="tetrad_gui")

    # Force import failure of nx_pydot so fallback kicks in
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == "networkx.drawing.nx_pydot":
            raise ImportError("simulated: nx_pydot missing")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    c = Controller(_DummyUI())
    G = nx.DiGraph(); G.add_edge("A", "B")
    pos = c._graphviz_pos(G)
    assert pos is None

    assert "GRAPHVIZ: nx_pydot not importable" in caplog.text
