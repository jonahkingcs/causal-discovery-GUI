# rthooks/jre_env.py
import os, sys, platform

base = getattr(sys, "_MEIPASS", None) or os.path.abspath(os.path.dirname(sys.argv[0]))

# candidate JRE locations inside the frozen app
candidates = []
if sys.platform == "darwin":
    # If running from .app, base is .../YourApp.app/Contents/MacOS
    parent   = os.path.abspath(os.path.join(base, os.pardir))
    grand    = os.path.abspath(os.path.join(parent, os.pardir))
    resources = os.path.join(grand, "Resources")
    candidates += [os.path.join(resources, "jre"), os.path.join(base, "jre")]
# fallbacks (e.g., running from a folder)
candidates += [os.path.join(base, "jre"), os.path.join(base, "runtime", "jre")]

jre_dir = next((p for p in candidates if os.path.isdir(p)), None)
if jre_dir:
    os.environ["JAVA_HOME"] = jre_dir
    lib_server = os.path.join(jre_dir, "lib", "server")
    if platform.system() == "Darwin":
        os.environ["DYLD_LIBRARY_PATH"] = os.pathsep.join(
            [lib_server, os.environ.get("DYLD_LIBRARY_PATH", "")])
    else:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            [lib_server, os.environ.get("LD_LIBRARY_PATH", "")])
    os.environ["PATH"] = os.pathsep.join([os.path.join(jre_dir, "bin"),
                                          os.environ.get("PATH", "")])
