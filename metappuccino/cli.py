import os, sys, subprocess
def main():
    base = os.environ.get("METAPPUCCINO_DIR", os.getcwd())
    script = os.path.join(base, "bin", "Metappuccino.py")
    if not os.path.isfile(script):
        print(f"missing: {script}", file=sys.stderr); return 2
    cmd = [sys.executable, script, *sys.argv[1:]]
    return subprocess.call(cmd)
if __name__ == "__main__":
    sys.exit(main())
