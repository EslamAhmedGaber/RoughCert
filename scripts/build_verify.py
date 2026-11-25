"""Compile both independent documents and run the four named QA gates."""
from pathlib import Path
import subprocess,json,sys,os,re
import fitz
ROOT=Path(__file__).resolve().parents[1]
M=ROOT/"manuscript"
A=ROOT/"audit"
def _qa_dir():
    # SWS_SCRIPTS overrides; otherwise the first installed copy of the writing-studio skill is used.
    candidates=[os.environ.get("SWS_SCRIPTS"), Path.home()/".codex/skills/scientific-writing-studio/scripts"]
    for c in candidates:
        if c and (Path(c)/"check_norms.py").is_file():return Path(c)
    raise FileNotFoundError("scientific-writing-studio scripts not found; set SWS_SCRIPTS")
QA=_qa_dir()
FUND="The authors acknowledge support from the Science, Technology & Innovation Funding Authority (STDF), Egypt, under the Basic Sciences Grant (Project ID 50925)."
def run(name,command,cwd):
    log=A/(name+".txt")
    env={**os.environ,"PYTHONUTF8":"1","PYTHONDONTWRITEBYTECODE":"1","MPLCONFIGDIR":str(A/"mpl")}
    with log.open("wb") as f:
        r=subprocess.run(command,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,env=env)
    print(name,r.returncode,flush=True)
    return dict(name=name,command=list(map(str,command)),cwd=str(cwd),exit=r.returncode,log=str(log.relative_to(ROOT)))
def main():
    A.mkdir(exist_ok=True)
    results=[]
    for doc,folder in [("main",M),("two_columns",M),("supplement",M/"supplement")]:
        for suffix in [".aux",".bbl",".blg",".log",".out"]:
            (folder/(doc+suffix)).unlink(missing_ok=True)
        tex=["pdflatex","-interaction=nonstopmode","-halt-on-error","-file-line-error",doc+".tex"]
        for i,cmd in enumerate([tex,["bibtex",doc],tex,tex]):
            step=run(doc+"_"+str(i+1),cmd,folder);results.append(step)
            if step["exit"]: break
        else:
            if "Label(s) may have changed" in (folder/(doc+".log")).read_text(errors="replace"):
                results.append(run(doc+"_stabilize",tex,folder))
            with fitz.open(folder/(doc+".pdf")) as pdf:results[-1]["pages"]=len(pdf)
    import re
    for stem,folder in [("main",M),("two_columns",M),("supplement",M/"supplement")]:
        log=(folder/(stem+".log")).read_text(errors="replace")
        pattern=r"Overfull \\[hv]box|Underfull \\hbox|undefined (?:references|citations)|LaTeX Warning: (?:Reference|Citation)|multiply defined|Label\(s\) may have changed|Missing character:|^!"
        issues=re.findall(pattern,log,re.I|re.M)
        results.append(dict(name=stem+"_log",exit=int(bool(issues)),issues=issues))
    for name,args,wd in [
        ("check_norms",["main.tex"],M),("check_tex_hygiene",["main.tex"],M),
        ("check_technical",["main.tex","--pdf","main.pdf","--bbl","main.bbl","--bib","references.bib","--order","refs-then-appendix","--declarations","before-refs","--funding",FUND,"--team-citation"],M),
        ("scan_leaks",["manuscript","--allow","*09_declarations.tex:(?i)ChatGPT and Claude"],ROOT)]:
        results.append(run(name,[sys.executable,"-B",str(QA/(name+".py")),*args],wd))
    (ROOT/"results/build_qa.json").write_text(json.dumps(results,indent=2))
    return max(r["exit"] for r in results)
if __name__=="__main__":sys.exit(main())

