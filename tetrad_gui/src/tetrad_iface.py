import importlib, jpype, jpype.imports, pandas as pd, networkx as nx
# tetrad wrapper
from pytetrad.tools.TetradSearch import TetradSearch
from jpype import JClass
import itertools, math
import numpy as np
from scipy.stats import chi2
Endpoint = JClass("edu.cmu.tetrad.graph.Endpoint")

# starts JVM if not already started
if not jpype.isJVMStarted():
    jpype.startJVM(convertStrings=False)

# fallback Java classes
ChiSquare = jpype.JClass("edu.cmu.tetrad.algcomparison.independence.ChiSquare")
BDeuScore = jpype.JClass("edu.cmu.tetrad.algcomparison.score.BdeuScore")
FisherZ   = jpype.JClass("edu.cmu.tetrad.algcomparison.independence.FisherZ")
SemBic    = jpype.JClass("edu.cmu.tetrad.algcomparison.score.SemBicScore")



def _mark_string(ep1: str, ep2: str) -> str:
    # map endpoint pairs to a readable ASCII mark
    # TAIL = plain line, ARROW = '>', CIRCLE = 'o'
    if ep1 == "TAIL" and ep2 == "ARROW":
        return "->"
    if ep1 == "ARROW" and ep2 == "TAIL":
        return "<-"
    if ep1 == "ARROW" and ep2 == "ARROW":
        return "<->"
    if ep1 == "CIRCLE" and ep2 == "ARROW":
        return "o->"
    if ep1 == "ARROW" and ep2 == "CIRCLE":
        return "<-o"
    if ep1 == "CIRCLE" and ep2 == "CIRCLE":
        return "o-o"
    if ep1 == "TAIL" and ep2 == "TAIL":
        return "---"
    # fallback
    return f"{ep1}|{ep2}"



# wrapper for loading data and running search algorithms
class TetradRunner:
    def __init__(self):
        self.sepsets = {}
        # java independence test used (fisher z or chi squared)
        self._last_test = None
        self._jnode_by_name = {}
        self._last_df = None
        self._dtype   = "auto"
        self._alpha   = 0.05

    # loads CSV files
    @staticmethod
    def load_dataframe(path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    # chooses the right statistical test for either discrete or continuous data
    def _attach_test_score(self, ts: TetradSearch, df: pd.DataFrame,
                           alpha: float, dtype: str
                           ):
        if dtype == "discrete":
            is_discrete = True
        elif dtype == "cont":
            is_discrete = False
        else:
            # checks numpy datatypes of all columns
            is_discrete = all(dt.kind in "iu" and df[c].nunique() <= 10 for c, dt in df.dtypes.items())

        def _maybe_call(name, *args):
            # checks if there is a class helper on the TetradSearch instance
            if hasattr(ts, name):                         
                return getattr(ts, name)(*args)
            # otherwise module helper is imported
            mod = importlib.import_module(
                'pytetrad.tools.TetradSearch')            
            if hasattr(mod, name):
                return getattr(mod, name)(ts, *args)
            raise AttributeError

        try:
            # if the helper classes were found
            if is_discrete:
                _maybe_call("use_chi_square", 1, alpha)
                _maybe_call("use_bdeu")
            else:
                _maybe_call("use_fisher_z", alpha)
                _maybe_call("use_sem_bic")
            return
        except AttributeError:
            # if the helpers were not found java classes attached directly
            if is_discrete:
                ts.TEST = ChiSquare()
                ts.SCORE = BDeuScore()
            else:
                ts.TEST = FisherZ()
                ts.SCORE = SemBic(True)



    # run PC or GFCI search and return a networkx digraph
    def run_search(self, df: pd.DataFrame, algo_name: str,
                alpha: float = 0.05, depth: int = -1, dtype: str = "auto", penalty_discount: float = 2.0
                ) -> nx.DiGraph:
        # create a new searcher on the dataset
        ts = TetradSearch(df)
        # adds correct test/score and significance level and data type
        self._attach_test_score(ts, df, alpha, dtype)
        self._last_df = df
        self._dtype = dtype
        self._alpha = alpha
        self._last_test = None
        self._jnode_by_name = {}

        self.sepsets = {}

        # debug: confirm tester and variable name
        try:
            t = self._last_test
            tname = type(t).__name__ if t else None
            talpha = t.getAlpha() if t and hasattr(t, "getAlpha") else None
            print(f"[run_search] tester={tname} alpha={talpha} domain={len(self._jnode_by_name)} vars")
        except Exception as e:
            print("[run_search] debug failed:", e)

        try:
            if self._last_test and hasattr(self._last_test, "getVariables"):
                jvars = list(self._last_test.getVariables())
                self._jnode_by_name = {str(v): v for v in jvars}
        except Exception:
            # if anything fails then leave mapping empty and skip sepsets
            pass

        # run the chosen algorithm
        algo = algo_name.upper()
        if algo == "PC":
            ts.run_pc(depth=depth)
            real_test = None

            # try common gettets on java search object
            for getter in ("getIndependenceTest", "getIndTest", "getTest"):
                try:
                    if hasattr(ts.java, getter):
                        real_test = getattr(ts.java, getter)()
                        if real_test is not None:
                            break
                except Exception:
                    pass

            # if none, try to build tester from java dataset
            if real_test is None:
                dataset = None
                for ds_getter in ("getData", "getDataSet", "getSourceGraph", "getSourceDataSet"):
                    try:
                        if hasattr(ts.java, ds_getter):
                            candidate = getattr(ts.java, ds_getter)()
                            if candidate is not None and hasattr(candidate, "getVariables"):
                                dataset = candidate
                                break
                    except Exception:
                        pass

                if dataset is not None:
                    try:
                        # choose tester by dtype
                        if dtype == "discrete":
                            IndTestChiSquare = JClass("edu.cmu.tetrad.search.IndTestChiSquare")
                            real_test = IndTestChiSquare(dataset, alpha)
                        else:
                            IndTestFisherZ = JClass("edu.cmu.tetrad.search.IndTestFisherZ")
                            real_test = IndTestFisherZ(dataset, alpha)
                    except Exception:
                        real_test = None

            # if we have a tester, build the name to node map
            try:
                if real_test is not None and hasattr(real_test, "getVariables"):
                    jvars = list(real_test.getVariables())
                    self._last_test = real_test
                    self._jnode_by_name = {str(v): v for v in jvars}
            except Exception:
                self._last_test = None
                self._jnode_by_name = {}

            # debug
            try:
                tname = type(self._last_test).__name__ if self._last_test else None
                talpha = self._last_test.getAlpha() if self._last_test and hasattr(self._last_test, "getAlpha") else None
                print(f"[run_search] tester={tname} alpha={talpha} domain={len(self._jnode_by_name)} vars")
            except Exception as e:
                print("[run_search] debug failed:", e)

        elif algo == "GFCI":
            ts.run_gfci(
                depth=depth,
                max_degree=-1,
                max_disc_path_length=-1,
                complete_rule_set_used=True,
                guarantee_pag=False,
            )

            real_test = None

            # try common gettets on java search object
            for getter in ("getIndependenceTest", "getIndTest", "getTest"):
                try:
                    if hasattr(ts.java, getter):
                        real_test = getattr(ts.java, getter)()
                        if real_test is not None:
                            break
                except Exception:
                    pass

            # if none, try to build tester from java dataset
            if real_test is None:
                dataset = None
                for ds_getter in ("getData", "getDataSet", "getSourceGraph", "getSourceDataSet"):
                    try:
                        if hasattr(ts.java, ds_getter):
                            candidate = getattr(ts.java, ds_getter)()
                            if candidate is not None and hasattr(candidate, "getVariables"):
                                dataset = candidate
                                break
                    except Exception:
                        pass

                if dataset is not None:
                    try:
                        # choose tester by dtype
                        if dtype == "discrete":
                            IndTestChiSquare = JClass("edu.cmu.tetrad.search.IndTestChiSquare")
                            real_test = IndTestChiSquare(dataset, alpha)
                        else:
                            IndTestFisherZ = JClass("edu.cmu.tetrad.search.IndTestFisherZ")
                            real_test = IndTestFisherZ(dataset, alpha)
                    except Exception:
                        real_test = None

            # if we have a tester, build the name to node map
            try:
                if real_test is not None and hasattr(real_test, "getVariables"):
                    jvars = list(real_test.getVariables())
                    self._last_test = real_test
                    self._jnode_by_name = {str(v): v for v in jvars}
            except Exception:
                self._last_test = None
                self._jnode_by_name = {}

            # debug
            try:
                tname = type(self._last_test).__name__ if self._last_test else None
                talpha = self._last_test.getAlpha() if self._last_test and hasattr(self._last_test, "getAlpha") else None
                print(f"[run_search] tester={tname} alpha={talpha} domain={len(self._jnode_by_name)} vars")
            except Exception as e:
                print("[run_search] debug failed:", e)

        elif algo == "FGES":
            # build searcher + choose score for the dtype (SEM-BIC for cont, BDeu for discrete)
            ts = TetradSearch(df)
            self._attach_test_score(ts, df, alpha, dtype)

            # set the BIC penalty on the Parameters object (used by FGES/SEM-BIC)
            try:
                p = float(penalty_discount)
                if hasattr(ts, "params") and hasattr(ts.params, "set"):
                    ts.params.set("penaltyDiscount", p)  # <- key name used by Tetrad
            except Exception as e:
                print("[FGES] couldn't set penaltyDiscount:", e)

            # run FGES; pass only the boolean by name (or fall back to positional)
            try:
                ts.run_fges(symmetric_first_step=False)
            except TypeError:
                ts.run_fges(False)

            # FGES is score-based (no Java independence tester to keep)
            self._last_test = None
            self._jnode_by_name = {}

        else:
            raise ValueError(f"Unknown algorithm {algo_name}")

        # convert Tetrad graph to NetworkX DiGraph
        g_nx = self._to_networkx(ts.java)
        return g_nx
    


    def _to_networkx(self, jgraph) -> nx.DiGraph:
        g = nx.DiGraph()
        for n in jgraph.getNodes():
            g.add_node(str(n))
        for e in jgraph.getEdges():
            a, b = str(e.getNode1()), str(e.getNode2())
            ep1, ep2 = str(e.getEndpoint1()), str(e.getEndpoint2())
            mark = _mark_string(ep1, ep2)
            # store one canonical edge a--b with endpoint attributes
            g.add_edge(a, b, ep1=ep1, ep2=ep2, mark=mark,
                    directed=(ep1 == "TAIL" and ep2 == "ARROW"))
        return g



    # try finding condition set S that says a ⟂ b | S
    def find_sepset(self, a_name: str, b_name: str, candidates, max_k: int = 3):
        """
        Try to find a conditioning set S ⊆ candidates with |S| ≤ max_k
        such that the CI test says a ⟂ b | S. Returns (S_names, p_value) or None.
        Includes debug prints so we can see what's happening.
        """
        test = self._last_test
        print(f"[find_sepset] A={a_name} B={b_name}  tester={'None' if test is None else type(test).__name__}")
        if test is None:
            return None

        # make sure tester uses alpha if stored
        try:
            if hasattr(self, "_alpha") and hasattr(test, "setAlpha"):
                test.setAlpha(self._alpha)
        except Exception:
            pass
        try:
            talpha = test.getAlpha() if hasattr(test, "getAlpha") else None
            print(f"[find_sepset] tester_alpha={talpha}")
        except Exception:
            print("[find_sepset] tester_alpha=?")

        # map variable name to Java node
        A = self._jnode_by_name.get(a_name)
        B = self._jnode_by_name.get(b_name)
        print(f"[find_sepset] A_in_domain={A is not None}  B_in_domain={B is not None}")
        if A is None or B is None:
            return None

        # build an initial pool from candidates provided
        pool_names = [n for n in candidates if n in self._jnode_by_name and n not in (a_name, b_name)]
        # fallback option, widen to all variables
        if len(pool_names) < 2:
            pool_names = [n for n in self._jnode_by_name.keys() if n not in (a_name, b_name)]
        print(f"[find_sepset] pool_names={len(pool_names)}  max_k={max_k}")

        pool = [self._jnode_by_name[n] for n in pool_names]
        ArrayList = JClass('java.util.ArrayList')

        tested = 0
        # try conditioning set sizes from zero to max_k
        for k in range(0, min(max_k, len(pool)) + 1):
            approx = math.comb(len(pool), k) if hasattr(math, "comb") else 0
            print(f"[find_sepset] trying k={k} (combos≈{approx})")
            for combo in itertools.combinations(pool, k):
                tested += 1
                Z = ArrayList()
                for z in combo:
                    Z.add(z)
                try:
                    # true => independent at tester's alpha
                    indep = test.isIndependent(A, B, Z)
                except Exception:
                    continue
                if indep:
                    # p-value if the test exposes it (FisherZ does)
                    p = None
                    try:
                        p = float(test.getPValue())
                    except Exception:
                        pass
                    S_names = [str(z) for z in combo]
                    print(f"[find_sepset] FOUND S={S_names} p={p}")
                    return (S_names, p)

        print(f"[find_sepset] tested {tested} combos; none independent at tester.alpha")
        return None
    

    
    def _effective_dtype(self, df):
        if self._dtype in ("cont", "discrete"):
            return self._dtype
        # auto: simple heuristic, if any float columns -> continuous
        # else if all int like with small cardinality -> discrete
        if any(np.issubdtype(dt, np.floating) for dt in df.dtypes):
            return "cont"
        unique_small = all((np.issubdtype(dt, np.integer) and df[c].nunique() <= 10)
                        for c, dt in df.dtypes.items())
        return "discrete" if unique_small else "cont"
    


    def find_sepset_py(self, a_name: str, b_name: str, candidates, max_k: int = 3):
        """
        Try to find S ⊆ candidates, |S| ≤ max_k, such that A ⟂ B | S.
        Returns (S_names, p_value) or None.
        Uses:
        - continuous: partial correlation (Fisher-Z)
        - discrete: conditional G^2 (chi-square) across strata of S
        """

        # uses the last loaded dataframe
        df = self._last_df
        if df is None or a_name not in df.columns or b_name not in df.columns:
            return None

        # picks appropriate datatype
        dtype = self._effective_dtype(df)
        # filter candidate variables to exclude A and B
        pool = [n for n in candidates if n in df.columns and n not in (a_name, b_name)]

        # brute force all subsets S
        for k in range(0, min(max_k, len(pool)) + 1):
            for S in itertools.combinations(pool, k):
                # runs appropriate conditional independence test
                if dtype == "cont":
                    ok, p = self._ci_continuous(df, a_name, b_name, list(S))
                else:
                    ok, p = self._ci_discrete(df, a_name, b_name, list(S))
                # returns first sepset that has a p value above alpha
                if ok:
                    return (list(S), p)
        return None



    def _ci_continuous(self, df, a, b, S):
        """
        Test A ⟂ B | S using partial correlation:
        - Regress A~S and B~S (OLS), take residuals rA, rB
        - Pearson r between rA and rB
        - Fisher-Z z = 0.5*ln((1+r)/(1-r)) * sqrt(n - |S| - 3)
        - Two-sided p via normal tail (erfc)
        """
        # convert A and B to float arrays
        yA = df[a].astype(float).values
        yB = df[b].astype(float).values
        n  = len(df)

        if S:
            X = np.column_stack([df[s].astype(float).values for s in S] + [np.ones(n)])
            # residuals
            betaA, *_ = np.linalg.lstsq(X, yA, rcond=None)
            betaB, *_ = np.linalg.lstsq(X, yB, rcond=None)
            rA = yA - X @ betaA
            rB = yB - X @ betaB
        else:
            rA, rB = yA - yA.mean(), yB - yB.mean()

        # pearson r
        denom = (np.linalg.norm(rA) * np.linalg.norm(rB))
        if denom == 0:
            return False, None
        r = float(np.dot(rA, rB) / denom)
        r = max(-0.999999, min(0.999999, r))

        dof = n - len(S) - 3
        # must have positive degrees of freedom for fisher-Z
        if dof <= 0:
            return False, None

        # fisher z transform to z statistic
        z  = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(dof)
        # two sided p from Normal(0,1) using erfc
        p  = math.erfc(abs(z) / math.sqrt(2.0))

        return (p >= self._alpha, p)
    


    # implements G squared independence test
    def _ci_discrete(self, df, a, b, S):
        """
        Test A ⟂ B | S for discrete variables with conditional G^2:
        Sum over strata of S: 2 * sum_{cells} n * ln(n * n_s / (n_as * n_bs))
        df = sum over strata of (|A|-1)(|B|-1)
        p = chi2.sf(G2, df)
        """

        A = df[a].astype(int)
        B = df[b].astype(int)
        if S:
            strata = df[S].astype(int)
        else:
            strata = None

        a_levels = np.unique(A)
        b_levels = np.unique(B)

        G2 = 0.0
        df_total = 0

        # computer G squared on the A x B contigency table
        if strata is None:
            # single 2D table
            G2, df_total = self._g2_table(A, B, a_levels, b_levels)
        else:
            # build a composite key for S tuple of values per row
            keys = list(zip(*[strata[s].values for s in S]))
            keys = np.array(keys, dtype=object)
            for key in np.unique(keys):
                mask = (keys == key)
                if mask.sum() == 0:
                    continue
                G2_s, df_s = self._g2_table(A[mask], B[mask], a_levels, b_levels)
                G2 += G2_s
                df_total += df_s

        if df_total <= 0:
            return False, None

        # turn the G squared test into a p value with chi squared revival function
        p = float(chi2.sf(G2, df_total))
        # return p
        return (p >= self._alpha, p)
    
    

    # build the A x B conntigency table counting occurences of each (a, b) pair
    def _g2_table(self, A, B, a_levels, b_levels):
        # contingency counts
        table = np.zeros((len(a_levels), len(b_levels)), dtype=float)
        a_map = {v:i for i,v in enumerate(a_levels)}
        b_map = {v:i for i,v in enumerate(b_levels)}
        for ai, bi in zip(A, B):
            table[a_map[ai], b_map[bi]] += 1.0

        # compute marginal totals and expected counts
        n = table.sum()
        if n == 0:
            return 0.0, 0

        row = table.sum(axis=1, keepdims=True)
        col = table.sum(axis=0, keepdims=True)
        exp = (row @ col) / n
        # G^2 with 0*log(0/q) = 0 convention
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(table > 0, table * np.log(table / np.maximum(exp, 1e-12)), 0.0)
        G2 = 2.0 * float(np.nansum(ratio))

        # degrees of freedom for a 2D independence test
        dof = (len(a_levels) - 1) * (len(b_levels) - 1)

        # return G squared statistic and degrees of freedom
        return G2, dof
