import os, json, re, random
import pandas as pd

random.seed(7)

TEMPLATE_JSON_A = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/data_templates_training/metadata_templates_test_withoutkeys.json"
TEMPLATE_CSV_B = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/all_cat_sets/create_synt_data/train_val.merged.tsv"
OUT_A = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/all_cat_sets/create_synt_data/testset_A.csv"
OUT_B = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/all_cat_sets/create_synt_data/testset_B.csv"

CLOSED = {
    "library_selection": ["polyA","inverse rRNA","hybrid selection","small RNA","other","unknown"],
    "sequencing_source": ["single-cell","bulk","spatial","unknown"],
    "biopsy_type": ["primary","metastasis","blood"],
    "sex": ["male","female","unknown"],
    "response": ["no treatment","unknown","stable","progressive","success"]
}

OPEN = {}
OPEN["organ"] = [
    "gallbladder","appendix","duodenum","jejunum","ileum","trachea","larynx","salivary gland","tongue","bladder",
    "ureter","urethra","fallopian tube","endometrium","pharynx","tonsil","spinal cord","meninges","bile duct","mesentery","peritoneum"
]
OPEN["cell_type"] = [
    "goblet cells","paneth cells","enteric neurons","urothelial cells","umbrella cells","smooth muscle cells","salivary acinar cells","taste receptor cells","laryngeal epithelial cells","tracheal ciliated cells",
    "tonsillar lymphocytes","mesothelial cells","schwann cells","oligodendrocytes","ependymal cells","basal urothelial cells","interstitial cells of cajal","ductal epithelial cells","follicular endometrial cells","mucosal epithelial cells","epithelial cells"
]
OPEN["cell_line"] = [
    "GBC-SD","H69","SCC-25","CAL 27","HSC-3","FaDu","Detroit 562","T24","RT4","UM-UC-3",
    "GBC-T","HT-1376","5637","SNU-1076","Ben-Men-1","NCI-H2052","SNG-M","HSG","OE-33D","EM-E6/E7/hTERT"
]
OPEN["biopsy_site"] = [
    "gallbladder","appendix","duodenum","jejunum","ileum","trachea","larynx","salivary gland","tongue","bladder",
    "ureter","urethra","fallopian tube","endometrium","pharynx","tonsil","spinal cord","meninges","bile duct","mesentery","peritoneum","urine","bile"
]
OPEN["disease"] = [
    "healthy","cholelithiasis","acute cholecystitis","appendicitis","duodenal ulcer","celiac disease","jejunitis","infectious ileitis","laryngeal squamous cell carcinoma","laryngitis","tracheitis",
    "urothelial carcinoma","acute cystitis","ureteral carcinoma","urethral carcinoma","endometrial carcinoma","mucoepidermoid carcinoma","tonsillitis","pharyngitis","spinal cord injury","meningitis",
    "cholangiocarcinoma","peritonitis","peritoneal mesothelioma","fallopian tube carcinoma"
]
OPEN["treatment"] = [
    "no treatment","cholecystectomy","appendectomy","proton pump inhibitor therapy","gluten-free diet","laryngectomy","tracheostomy","intravesical BCG","transurethral resection","hysteroscopic curettage","salpingectomy",
    "tonsillectomy","broad-spectrum antibiotics","spinal decompression surgery","meningitis antimicrobial therapy","ERCP with stenting","peritoneal lavage","platinum-based chemotherapy","IMRT radiotherapy","mitomycin intravesical","endoscopic mucosal resection",
    "biliary drainage","ureteral stenting","urethroplasty","hormonal therapy","unknown"
]
OPEN["treatment_time"] = [
    "8 hours","16 hours","12 hours","3 days","7 days","14 days","21 days","28 days","2 months","3 months",
    "9 months","12 months","baseline","cycle 1 day 1","end of treatment","follow-up 30 days","follow-up 90 days","pre-operative","post-operative day 3","post-operative day 10","unknown"
]
OPEN["age"] = [f"{x} years" for x in [18,21,23,27,31,34,37,39,42,45,48,51,54,57,60,63,66,69,72,75,80,85]] + ["unknown"]
OPEN["ethnicity"] = [
    "Central European","West African","East African","Horn of Africa","North American","Central American","South American","Caribbean Creole","Andean Highlander","Amazonian",
    "Central Asian","Siberian","Caucasus","Levantine","Maghrebi","Iberian","Nordic-Baltic","Balkan","Malay Archipelago","Melanesian","Micronesian","Aotearoa Polynesian","Aboriginal Australian","unknown"
]

UNK_PCT = {
    "library_selection": 0.08,
    "sequencing_source": 0.08,
    "biopsy_site": 0.06,
    "biopsy_type": 0.05,
    "cell_line": 0.01,
    "cell_type": 0.05,
    "organ": 0.03,
    "disease": 0.06,
    "treatment": 0.01,
    "treatment_time": 0.01,
    "response": 0.01,
    "age": 0.08,
    "sex": 0.05,
    "ethnicity": 0.07
}

MANUAL_SYNONYMS = {
    "library_selection": {
        "polyA": ["polyA", "poly-A", "polyadenylated", "polyA+ selection", "oligo-dT selection"],
        "inverse rRNA": ["rRNA depletion", "ribosomal RNA removal", "ribo-minus", "rRNA-depleted", "rRNA−"],
        "hybrid selection": ["hybrid capture", "capture-based selection", "target enrichment", "probe capture", "hyb-capture"],
        "small RNA": ["small-RNA", "smRNA", "miRNA-enriched", "short RNA fraction", "microRNA prep"],
        "other": ["other selection", "misc selection", "non-standard selection", "custom selection", "unspecified selection"],
        "unknown": ["unknown selection", "selection n/a", "selection not reported", "selection unspecified", "undisclosed selection"]
    },
    "sequencing_source": {
        "single-cell": ["single cell", "scRNA-seq", "single-cell profiling", "cellular-level capture", "single-cell assay"],
        "bulk": ["bulk RNA-seq", "pooled tissue", "bulk profiling", "population average", "tissue-level assay"],
        "spatial": ["spatial transcriptomics", "spatially resolved", "in situ capture", "slide-based spatial", "ST assay"],
        "unknown": ["unknown source", "source n/a", "source not reported", "unspecified source", "undisclosed source"]
    },
    "biopsy_type": {
        "primary": ["primary lesion", "index tumor", "native tissue", "de novo site", "primary site"],
        "metastasis": ["metastatic lesion", "secondary site", "metastatic focus", "distant metastasis", "met site"],
        "blood": ["peripheral blood", "blood draw", "venous blood", "whole blood", "blood specimen"]
    },
    "sex": {
        "male": ["male", "M", "man", "masculine", "biological male"],
        "female": ["female", "F", "woman", "feminine", "biological female"],
        "unknown": ["sex unknown", "sex n/a", "sex not reported", "unspecified sex", "undisclosed sex"]
    },
    "response": {
        "no treatment": ["no treatment", "untreated", "treatment-naïve", "no therapy given", "observation only"],
        "unknown": ["response unknown", "response n/a", "response not reported", "unspecified response", "undetermined response"],
        "stable": ["stable disease", "no change", "disease stability", "SD", "unchanged status"],
        "progressive": ["progressive disease", "PD", "worsening", "disease progression", "progression"],
        "success": ["complete response", "CR", "resolved", "marked improvement", "therapeutic success"]
    },
    "organ": {
        "gallbladder": ["gallbladder", "cholecyst", "GB", "biliary sac", "gall bladder"],
        "appendix": ["appendix", "vermiform appendix", "appendiceal", "appendicular", "appendiceal organ"],
        "duodenum": ["duodenum", "duodenal", "proximal small intestine", "first part of small bowel", "D1–D2"],
        "jejunum": ["jejunum", "jejunal", "mid small intestine", "small bowel middle", "D3–D4/jejunum"],
        "ileum": ["ileum", "ileal", "distal small intestine", "terminal ileum", "small bowel distal"],
        "trachea": ["trachea", "windpipe", "tracheal", "airway trachea", "proximal airway"],
        "larynx": ["larynx", "voice box", "laryngeal", "glottic region", "supraglottic area"],
        "salivary gland": ["salivary gland", "major salivary", "parotid/submandibular", "salivary tissue", "salivary organ"],
        "tongue": ["tongue", "lingual", "oral tongue", "glossal", "lingua"],
        "bladder": ["urinary bladder", "bladder", "vesical", "urovesical", "UB"],
        "ureter": ["ureter", "ureteric", "upper urinary tract", "ureteral", "renal pelvis outflow"],
        "urethra": ["urethra", "urethral", "lower urinary tract", "urinary outlet", "urethral canal"],
        "fallopian tube": ["fallopian tube", "uterine tube", "oviduct", "FT", "salpinx"],
        "endometrium": ["endometrium", "uterine lining", "endometrial", "uterine mucosa", "uterine endometrium"],
        "pharynx": ["pharynx", "pharyngeal", "throat", "oro/nasopharynx", "hypopharynx"],
        "tonsil": ["tonsil", "palatine tonsil", "tonsillar", "Waldeyer’s ring", "tonsillar tissue"],
        "spinal cord": ["spinal cord", "medulla spinalis", "spinal parenchyma", "cord tissue", "SC"],
        "meninges": ["meninges", "meningeal", "dura/arachnoid/pia", "meningeal layers", "meninx"],
        "bile duct": ["bile duct", "biliary duct", "common bile duct", "CBD", "intrahepatic duct"],
        "mesentery": ["mesentery", "mesenteric tissue", "mesenterium", "intestinal mesentery", "mesenteric fold"],
        "peritoneum": ["peritoneum", "peritoneal", "serosal cavity", "peritoneal lining", "peritoneal membrane"]
    },
    "cell_type": {
        "goblet cells": ["goblet cells", "mucus-secreting cells", "goblet cell type", "intestinal goblet", "goblet epithelium"],
        "paneth cells": ["paneth cells", "defensin-secreting cells", "paneth lineage", "crypt paneth", "paneth epithelium"],
        "enteric neurons": ["enteric neurons", "ENS neurons", "myenteric neurons", "enteric nervous cells", "Auerbach plexus neurons"],
        "urothelial cells": ["urothelial cells", "transitional epithelium", "urothelium", "bladder lining cells", "urothelial epithelium"],
        "umbrella cells": ["umbrella cells", "superficial urothelial", "umbrella layer", "urothelial surface cells", "apical urothelium"],
        "smooth muscle cells": ["smooth muscle cells", "SMCs", "visceral myocytes", "smooth muscle lineage", "myocytic cells"],
        "salivary acinar cells": ["salivary acinar cells", "acinar secretory cells", "salivary acini", "serous/mucous acini", "acinar epithelium"],
        "taste receptor cells": ["taste receptor cells", "gustatory receptors", "taste bud cells", "chemosensory cells", "TRCs"],
        "laryngeal epithelial cells": ["laryngeal epithelial cells", "laryngeal epithelium", "larynx mucosa cells", "vocal fold epithelium", "laryngeal lining cells"],
        "tracheal ciliated cells": ["tracheal ciliated cells", "airway ciliated epithelium", "ciliated tracheal cells", "mucociliary cells", "tracheal epithelial ciliated"],
        "tonsillar lymphocytes": ["tonsillar lymphocytes", "tonsil lymphoid cells", "tonsillar T/B cells", "lymphocytes from tonsil", "tonsillar immune cells"],
        "mesothelial cells": ["mesothelial cells", "peritoneal mesothelium", "serosal mesothelium", "mesothelial lining", "mesothelium cells"],
        "schwann cells": ["schwann cells", "peripheral glia", "neurilemmocytes", "myelinating glia PNS", "schwann lineage"],
        "oligodendrocytes": ["oligodendrocytes", "central myelinating glia", "OLG cells", "oligodendroglia", "CNS myelin cells"],
        "ependymal cells": ["ependymal cells", "ependymocytes", "ventricular lining cells", "central canal epithelium", "CNS ependyma"],
        "basal urothelial cells": ["basal urothelial cells", "urothelial basal layer", "basal urothelium", "bladder basal cells", "urothelial progenitors"],
        "interstitial cells of cajal": ["interstitial cells of Cajal", "ICC", "gut pacemaker cells", "Cajal cells", "GI pacemaker"],
        "ductal epithelial cells": ["ductal epithelial cells", "duct epithelium", "biliary duct cells", "salivary duct epithelium", "ductal lining cells"],
        "follicular endometrial cells": ["follicular endometrial cells", "endometrial stromal/follicular", "uterine follicular cells", "endometrium follicular", "follicular uterine cells"],
        "mucosal epithelial cells": ["mucosal epithelial cells", "mucosa epithelium", "surface mucosal cells", "mucosal lining", "mucosa-derived epithelium"],
        "epithelial cells": ["epithelial cells", "epithelium", "parenchymal epithelial", "lining epithelial cells", "surface epithelium"]
    },
    "cell_line": {
        "GBC-SD": ["GBC-SD", "GBC SD", "gallbladder cancer line GBC-SD", "GBCSD", "GBC-SD cell line"],
        "H69": ["H69", "cholangiocarcinoma line H69", "H-69", "H69 cells", "biliary H69"],
        "SCC-25": ["SCC-25", "SCC25", "tongue SCC-25", "oral SCC25", "SCC 25 cells"],
        "CAL 27": ["CAL 27", "CAL-27", "oral cancer CAL27", "CAL27 line", "CAL_27"],
        "HSC-3": ["HSC-3", "HSC3", "tongue HSC-3", "HSC 3 cells", "oral SCC HSC-3"],
        "FaDu": ["FaDu", "pharyngeal FaDu", "FaDu line", "FaDu cells", "hypopharyngeal FaDu"],
        "Detroit 562": ["Detroit 562", "Detroit-562", "pharynx Detroit562", "DET562", "Detroit562 cells"],
        "T24": ["T24", "bladder T24", "T-24 cells", "T24 urothelial", "T24 line"],
        "RT4": ["RT4", "bladder RT4", "RT-4 cells", "RT4 urothelial", "RT4 line"],
        "UM-UC-3": ["UM-UC-3", "UMUC3", "UM-UC3", "UM UC 3", "UM-UC-3 cells"],
        "GBC-T": ["GBC-T", "GBCT", "gallbladder line GBC-T", "GBC T", "GBC-T cells"],
        "HT-1376": ["HT-1376", "HT1376", "urothelial HT-1376", "HT 1376 cells", "HT-1376 line"],
        "5637": ["5637", "urothelial 5637", "5637 cells", "5637 line", "bladder 5637"],
        "SNU-1076": ["SNU-1076", "SNU1076", "laryngeal SNU-1076", "SNU 1076 cells", "SNU-1076 line"],
        "Ben-Men-1": ["Ben-Men-1", "BenMen1", "meningioma Ben-Men-1", "Ben Men 1", "Ben-Men1"],
        "NCI-H2052": ["NCI-H2052", "H2052", "mesothelioma H2052", "NCI H2052", "H-2052 cells"],
        "SNG-M": ["SNG-M", "SNGM", "mesentery SNG-M", "SNG M", "SNG-M line"],
        "HSG": ["HSG", "salivary HSG", "HSG cells", "HSG line", "HSG salivary gland"],
        "OE-33D": ["OE-33D", "OE33D", "duodenum OE-33D", "OE 33D", "OE-33D line"],
        "EM-E6/E7/hTERT": ["EM-E6/E7/hTERT", "EM E6 E7 hTERT", "endometrial EM-E6/E7/hTERT", "EM-E6E7-hTERT", "EM-E6/E7/hTERT cells"]
    },
    "biopsy_site": {
        "urine": ["urine", "urinary sample", "voided urine", "urine specimen", "urinary fluid"],
        "bile": ["bile", "biliary fluid", "gall bile", "bile specimen", "biliary sample"],
        "gallbladder": ["gallbladder site", "GB site", "gall bladder site", "gallbladder tissue site", "cholecyst site"],
        "appendix": ["appendix site", "appendiceal site", "appendicular site", "vermiform appendix site", "appendix tissue site"],
        "duodenum": ["duodenum site", "duodenal site", "duodenal tissue site", "D1/D2 site", "proximal small bowel site"],
        "jejunum": ["jejunum site", "jejunal site", "mid small bowel site", "jejunum tissue site", "jejunal sampling"],
        "ileum": ["ileum site", "ileal site", "distal small bowel site", "terminal ileum site", "ileal sampling"],
        "trachea": ["trachea site", "tracheal site", "airway trachea site", "tracheal sampling", "proximal airway site"],
        "larynx": ["larynx site", "laryngeal site", "laryngeal sampling", "voice box site", "glottic region site"],
        "salivary gland": ["salivary gland site", "salivary sampling", "parotid/submandibular site", "salivary tissue site", "salivary organ site"],
        "tongue": ["tongue site", "lingual site", "oral tongue site", "glossal site", "lingua site"],
        "bladder": ["bladder site", "vesical site", "urinary bladder site", "UB site", "bladder tissue site"],
        "ureter": ["ureter site", "ureteric site", "upper tract site", "ureter sampling", "ureteral site"],
        "urethra": ["urethra site", "urethral site", "lower tract site", "urethral sampling", "urethra tissue site"],
        "fallopian tube": ["fallopian tube site", "FT site", "oviduct site", "uterine tube site", "salpinx site"],
        "endometrium": ["endometrium site", "uterine lining site", "endometrial tissue site", "uterine mucosa site", "endometrium sampling"],
        "pharynx": ["pharynx site", "pharyngeal site", "throat site", "oro/hypopharynx site", "pharyngeal sampling"],
        "tonsil": ["tonsil site", "tonsillar site", "palatine tonsil site", "tonsil sampling", "Waldeyer ring site"],
        "spinal cord": ["spinal cord site", "cord site", "SC sampling", "spinal parenchyma site", "medulla spinalis site"],
        "meninges": ["meninges site", "meningeal site", "dura/arachnoid/pia site", "meningeal sampling", "meninx site"],
        "bile duct": ["bile duct site", "CBD site", "biliary duct site", "bile duct sampling", "intrahepatic duct site"],
        "mesentery": ["mesentery site", "mesenteric site", "mesenterium site", "intestinal mesentery site", "mesenteric sampling"],
        "peritoneum": ["peritoneum site", "peritoneal site", "serosal site", "peritoneal lining site", "peritoneal sampling"]
    },
    "disease": {
        "healthy": ["healthy", "no disease", "normal", "asymptomatic", "clinically normal"],
        "cholelithiasis": ["cholelithiasis", "gallstones", "biliary calculi", "gall stone disease", "cholelithic disease"],
        "acute cholecystitis": ["acute cholecystitis", "inflamed gallbladder", "acute GB inflammation", "acute biliary cholecystitis", "GB inflammation acute"],
        "appendicitis": ["appendicitis", "inflamed appendix", "acute appendicitis", "appendiceal inflammation", "appendicular inflammation"],
        "duodenal ulcer": ["duodenal ulcer", "peptic ulcer of duodenum", "duodenal peptic ulcer", "DU", "ulcer in duodenum"],
        "celiac disease": ["celiac disease", "coeliac disease", "gluten-sensitive enteropathy", "celiac sprue", "coeliac sprue"],
        "jejunitis": ["jejunitis", "jejunal inflammation", "inflamed jejunum", "enteritis of jejunum", "jejunal enteritis"],
        "infectious ileitis": ["infectious ileitis", "ileal infection", "ileitis infectious", "infected ileum", "infective ileitis"],
        "laryngeal squamous cell carcinoma": ["laryngeal squamous cell carcinoma", "laryngeal SCC", "larynx SCC", "LSCC", "squamous carcinoma of larynx"],
        "laryngitis": ["laryngitis", "inflamed larynx", "laryngeal inflammation", "acute laryngitis", "laryngeal irritation"],
        "tracheitis": ["tracheitis", "inflamed trachea", "tracheal inflammation", "airway tracheitis", "acute tracheitis"],
        "urothelial carcinoma": ["urothelial carcinoma", "transitional cell carcinoma", "TCC", "bladder cancer (urothelial)", "urothelial cancer"],
        "acute cystitis": ["acute cystitis", "bladder infection", "acute UTI bladder", "acute vesical cystitis", "cystitis acute"],
        "ureteral carcinoma": ["ureteral carcinoma", "ureter cancer", "upper tract urothelial carcinoma", "UTUC ureter", "ureteral UC"],
        "urethral carcinoma": ["urethral carcinoma", "urethra cancer", "urothelial carcinoma of urethra", "urethral UC", "cancer of urethra"],
        "endometrial carcinoma": ["endometrial carcinoma", "uterine endometrial cancer", "endometrial cancer", "EC", "carcinoma of endometrium"],
        "mucoepidermoid carcinoma": ["mucoepidermoid carcinoma", "MEC", "salivary MEC", "muco-epidermoid cancer", "mucoepidermoid CA"],
        "tonsillitis": ["tonsillitis", "inflamed tonsil", "tonsillar inflammation", "acute tonsillitis", "tonsil infection"],
        "pharyngitis": ["pharyngitis", "sore throat", "pharyngeal inflammation", "acute pharyngitis", "throat inflammation"],
        "spinal cord injury": ["spinal cord injury", "SCI", "injury to spinal cord", "cord trauma", "spinal injury"],
        "meningitis": ["meningitis", "meningeal infection", "CNS meningitis", "infectious meningitis", "meninx inflammation"],
        "cholangiocarcinoma": ["cholangiocarcinoma", "bile duct cancer", "CCA", "cholangiocarcinoma of CBD", "biliary tract carcinoma"],
        "peritonitis": ["peritonitis", "peritoneal inflammation", "inflamed peritoneum", "acute peritonitis", "infectious peritonitis"],
        "peritoneal mesothelioma": ["peritoneal mesothelioma", "peritoneal MPM", "mesothelioma of peritoneum", "peritoneal malignant mesothelioma", "abdominal mesothelioma"],
        "fallopian tube carcinoma": ["fallopian tube carcinoma", "FTC", "oviduct carcinoma", "tubal carcinoma", "uterine tube cancer"]
    },
    "treatment": {
        "no treatment": ["no treatment", "untreated", "no therapy given", "watchful waiting", "observation only"],
        "cholecystectomy": ["cholecystectomy", "gallbladder removal", "lap chole", "open cholecystectomy", "chole surgery"],
        "appendectomy": ["appendectomy", "appendix removal", "appendicectomy", "lap appendectomy", "appendix surgery"],
        "proton pump inhibitor therapy": ["proton pump inhibitor therapy", "PPI therapy", "omeprazole/esomeprazole", "acid suppression therapy", "PPI course"],
        "gluten-free diet": ["gluten-free diet", "GFD", "gluten restriction", "celiac diet", "gluten elimination diet"],
        "laryngectomy": ["laryngectomy", "larynx removal", "partial laryngectomy", "total laryngectomy", "laryngeal resection"],
        "tracheostomy": ["tracheostomy", "trach", "tracheal stoma", "tracheotomy", "tracheostoma"],
        "intravesical BCG": ["intravesical BCG", "BCG instillation", "bladder BCG", "intravesical immunotherapy", "BCG therapy"],
        "transurethral resection": ["transurethral resection", "TUR", "TURBT", "endoscopic resection", "transurethral tumor resection"],
        "hysteroscopic curettage": ["hysteroscopic curettage", "D&C hysteroscopic", "uterine curettage", "hysteroscopy with curettage", "endometrial curettage"],
        "salpingectomy": ["salpingectomy", "fallopian tube removal", "tubal resection", "salpingeal resection", "FT excision"],
        "tonsillectomy": ["tonsillectomy", "tonsil removal", "tonsil surgery", "excision of tonsils", "tonsillar excision"],
        "broad-spectrum antibiotics": ["broad-spectrum antibiotics", "empiric antibiotics", "broad antibiotics", "broad-spectrum abx", "empirical antimicrobial therapy"],
        "spinal decompression surgery": ["spinal decompression surgery", "decompression of spinal cord", "spine decompression", "laminectomy decompression", "cord decompression"],
        "meningitis antimicrobial therapy": ["meningitis antimicrobial therapy", "meningitis antibiotics", "antimicrobial regimen for meningitis", "CNS infection therapy", "meningitis-targeted antibiotics"],
        "ERCP with stenting": ["ERCP with stenting", "endoscopic biliary stent", "biliary ERCP stent", "ERCP stent placement", "endoscopic stent biliary"],
        "peritoneal lavage": ["peritoneal lavage", "peritoneal washout", "abdominal lavage", "peritoneal irrigation", "washout of peritoneum"],
        "platinum-based chemotherapy": ["platinum-based chemotherapy", "platinum chemo", "cisplatin/carboplatin regimen", "platinum doublet", "platinum-containing therapy"],
        "IMRT radiotherapy": ["IMRT radiotherapy", "IMRT", "intensity-modulated radiotherapy", "modulated radiation therapy", "IMRT radiation"],
        "mitomycin intravesical": ["mitomycin intravesical", "intravesical mitomycin", "bladder mitomycin", "MMC instillation", "mitomycin C intravesical"],
        "endoscopic mucosal resection": ["endoscopic mucosal resection", "EMR", "mucosal resection endoscopic", "endoscopic removal of mucosa", "EMR procedure"],
        "biliary drainage": ["biliary drainage", "bile duct drainage", "external/internal drainage", "biliary decompression", "drainage of biliary tree"],
        "ureteral stenting": ["ureteral stenting", "ureteric stent", "double-J stent", "JJ stent placement", "ureteral stent placement"],
        "urethroplasty": ["urethroplasty", "urethral reconstruction", "urethral repair", "urethra plasty", "reconstructive urethral surgery"],
        "hormonal therapy": ["hormonal therapy", "endocrine therapy", "hormone treatment", "HT", "anti-hormonal regimen"],
        "unknown": ["treatment unknown", "therapy n/a", "treatment not reported", "unspecified therapy", "undisclosed treatment"]
    },
    "treatment_time": {
        "8 hours": ["8 hours", "8h", "at 8 hours", "timepoint 8h", "8-hr"],
        "16 hours": ["16 hours", "16h", "at 16 hours", "timepoint 16h", "16-hr"],
        "12 hours": ["12 hours", "12h", "at 12 hours", "timepoint 12h", "12-hr"],
        "3 days": ["3 days", "day 3", "at 3 days", "72 hours", "D3"],
        "7 days": ["7 days", "day 7", "at 7 days", "1 week", "D7"],
        "14 days": ["14 days", "day 14", "at 14 days", "2 weeks", "D14"],
        "21 days": ["21 days", "day 21", "at 21 days", "3 weeks", "D21"],
        "28 days": ["28 days", "day 28", "at 28 days", "4 weeks", "D28"],
        "2 months": ["2 months", "at 2 months", "M2", "approx. 60 days", "two-month mark"],
        "3 months": ["3 months", "at 3 months", "M3", "approx. 90 days", "three-month mark"],
        "9 months": ["9 months", "at 9 months", "M9", "approx. 270 days", "nine-month mark"],
        "12 months": ["12 months", "at 12 months", "M12", "one-year mark", "twelve months"],
        "baseline": ["baseline", "pre-treatment baseline", "initial visit", "time zero", "pretreatment baseline"],
        "cycle 1 day 1": ["cycle 1 day 1", "C1D1", "start of cycle 1", "cycle1 day1", "first dosing day"],
        "end of treatment": ["end of treatment", "EOT", "treatment completion", "therapy end", "end-of-therapy"],
        "follow-up 30 days": ["follow-up 30 days", "FU30", "30-day follow-up", "post-therapy day 30", "1-month follow-up"],
        "follow-up 90 days": ["follow-up 90 days", "FU90", "90-day follow-up", "post-therapy day 90", "3-month follow-up"],
        "pre-operative": ["pre-operative", "preop", "pre-operative timepoint", "before surgery", "pre-surgical"],
        "post-operative day 3": ["post-operative day 3", "POD3", "day 3 after surgery", "postop day 3", "3 days post-op"],
        "post-operative day 10": ["post-operative day 10", "POD10", "day 10 after surgery", "postop day 10", "10 days post-op"],
        "unknown": ["time unknown", "timepoint n/a", "time not reported", "unspecified time", "undetermined time"]
    },
    "age": {
        "18 years": ["18 years", "18 yrs", "age 18", "18yo", "eighteen years"],
        "21 years": ["21 years", "21 yrs", "age 21", "21yo", "twenty-one years"],
        "23 years": ["23 years", "23 yrs", "age 23", "23yo", "twenty-three years"],
        "27 years": ["27 years", "27 yrs", "age 27", "27yo", "twenty-seven years"],
        "31 years": ["31 years", "31 yrs", "age 31", "31yo", "thirty-one years"],
        "34 years": ["34 years", "34 yrs", "age 34", "34yo", "thirty-four years"],
        "37 years": ["37 years", "37 yrs", "age 37", "37yo", "thirty-seven years"],
        "39 years": ["39 years", "39 yrs", "age 39", "39yo", "thirty-nine years"],
        "42 years": ["42 years", "42 yrs", "age 42", "42yo", "forty-two years"],
        "45 years": ["45 years", "45 yrs", "age 45", "45yo", "forty-five years"],
        "48 years": ["48 years", "48 yrs", "age 48", "48yo", "forty-eight years"],
        "51 years": ["51 years", "51 yrs", "age 51", "51yo", "fifty-one years"],
        "54 years": ["54 years", "54 yrs", "age 54", "54yo", "fifty-four years"],
        "57 years": ["57 years", "57 yrs", "age 57", "57yo", "fifty-seven years"],
        "60 years": ["60 years", "60 yrs", "age 60", "60yo", "sixty years"],
        "63 years": ["63 years", "63 yrs", "age 63", "63yo", "sixty-three years"],
        "66 years": ["66 years", "66 yrs", "age 66", "66yo", "sixty-six years"],
        "69 years": ["69 years", "69 yrs", "age 69", "69yo", "sixty-nine years"],
        "72 years": ["72 years", "72 yrs", "age 72", "72yo", "seventy-two years"],
        "75 years": ["75 years", "75 yrs", "age 75", "75yo", "seventy-five years"],
        "80 years": ["80 years", "80 yrs", "age 80", "80yo", "eighty years"],
        "85 years": ["85 years", "85 yrs", "age 85", "85yo", "eighty-five years"],
        "unknown": ["age unknown", "age n/a", "age not reported", "unspecified age", "undisclosed age"]
    },
    "ethnicity": {
        "Central European": ["Central European", "Central Europe", "CEU", "central EU", "central European ancestry"],
        "West African": ["West African", "W African", "western Africa", "West-Africa ancestry", "WAfr"],
        "East African": ["East African", "E African", "eastern Africa", "East-Africa ancestry", "EAfr"],
        "Horn of Africa": ["Horn of Africa", "Horn-African", "HoA ancestry", "northeast Africa", "Horn region"],
        "North American": ["North American", "N American", "North America", "NA ancestry", "North-Am"],
        "Central American": ["Central American", "C American", "Central America", "CA ancestry", "Central-Am"],
        "South American": ["South American", "S American", "South America", "SA ancestry", "South-Am"],
        "Caribbean Creole": ["Caribbean Creole", "Caribbean", "Creole Caribbean", "Carib-Creole", "Caribbean ancestry"],
        "Andean Highlander": ["Andean Highlander", "Andean", "High Andes ancestry", "Andean highlands", "Andean origin"],
        "Amazonian": ["Amazonian", "Amazon basin ancestry", "Amazon region", "Amazon-native", "Amazonia"],
        "Central Asian": ["Central Asian", "C Asian", "Central Asia", "CA Asia ancestry", "Central-Asia"],
        "Siberian": ["Siberian", "Siberia", "northern Asian ancestry", "Siberian origin", "Sib ancestry"],
        "Caucasus": ["Caucasus", "Caucasian region", "Caucasus ancestry", "Caucasus origin", "from Caucasus"],
        "Levantine": ["Levantine", "Levant region", "eastern Mediterranean ancestry", "Levant origin", "Levant"],
        "Maghrebi": ["Maghrebi", "Maghreb region", "Northwest African", "Maghribi", "Maghreb ancestry"],
        "Iberian": ["Iberian", "Iberia", "Iberian Peninsula ancestry", "SW European Iberian", "Iberian origin"],
        "Nordic-Baltic": ["Nordic-Baltic", "Nordic Baltic", "N-B European", "Scandinavian/Baltic", "NB ancestry"],
        "Balkan": ["Balkan", "Balkans", "SE European Balkan", "Balkan ancestry", "Balkan origin"],
        "Malay Archipelago": ["Malay Archipelago", "Malay Isles", "Maritime SE Asia", "Malay ancestry", "Malay region"],
        "Melanesian": ["Melanesian", "Melanesia", "Pacific Melanesian", "Melanesian ancestry", "Melanesian origin"],
        "Micronesian": ["Micronesian", "Micronesia", "Pacific Micronesian", "Micronesian ancestry", "Micronesian origin"],
        "Aotearoa Polynesian": ["Aotearoa Polynesian", "Polynesian (NZ)", "Aotearoa origin", "NZ Polynesian", "Aotearoa ancestry"],
        "Aboriginal Australian": ["Aboriginal Australian", "Indigenous Australian", "Aboriginal AU", "First Australians", "Aboriginal ancestry"],
        "unknown": ["ethnicity unknown", "ethnicity n/a", "ethnicity not reported", "unspecified ethnicity", "undisclosed ethnicity"]
    }
}

def _fuzzy_variants(x):
    s = str(x)
    base = [s, s.lower(), s.title(), s.replace(" ", "_"), s.replace(" ", "-"), s.replace("-", " "), s.replace("_", " ")]
    abbr = "".join([w[0] for w in re.split(r"[ _-]+", s) if w]) if len(s) <= 40 else ""
    if abbr and len(abbr) >= 2:
        base.append(abbr.upper()); base.append(abbr.lower())
    if re.search(r"\d", s):
        base.append(re.sub(r"\d+", lambda m: m.group(0)+"+", s))
    if " cells" in s:
        base.append(s.replace(" cells"," cell")); base.append(s.replace(" cells",""))
    if "carcinoma" in s:
        base.append(s.replace("carcinoma","CA")); base.append(s.replace("carcinoma","cancer"))
    if "therapy" in s:
        base.append(s.replace(" therapy"," tx")); base.append("tx "+s.replace(" therapy",""))
    if any(k in s for k in ["follow-up","post-operative","pre-operative"]):
        base.append(s.replace(" ",""))
    base = [v.strip() for v in base if v.strip()]
    return list(dict.fromkeys(base))

def ensure_minimum_synonyms(val, manual_list, minimum=5):
    pool = []
    if manual_list:
        pool.extend(manual_list)
    fuzz = _fuzzy_variants(val)
    for v in fuzz:
        if v not in pool:
            pool.append(v)
        if len(pool) >= max(minimum, 5):
            break
    while len(pool) < minimum:
        pool.append(val)
    return pool

def make_synonyms(OPEN, CLOSED, minimum=5):
    syn = {}
    for cat, values in OPEN.items():
        for v in values:
            m = MANUAL_SYNONYMS.get(cat, {}).get(v, [])
            syn[v] = ensure_minimum_synonyms(v, m, minimum=minimum)
    for cat in ["library_selection","sequencing_source","biopsy_type","sex","response"]:
        for v in CLOSED.get(cat, []):
            m = MANUAL_SYNONYMS.get(cat, {}).get(v, [])
            syn[v] = ensure_minimum_synonyms(v, m, minimum=minimum)
    return syn

SYN = make_synonyms(OPEN, CLOSED, minimum=5)

UNK_PCT = {
    "library_selection": 0.03,
    "sequencing_source": 0.03,
    "biopsy_site": 0.02,
    "biopsy_type": 0.02,
    "cell_line": 0.05,
    "cell_type": 0.04,
    "organ": 0.01,
    "disease": 0.02,
    "treatment": 0.04,
    "treatment_time": 0.06,
    "response": 0.06,
    "age": 0.03,
    "sex": 0.02,
    "ethnicity": 0.03
}

def _should_set_unknown(key, counts):
    target = UNK_PCT.get(key, 0.0)
    if target <= 0.0:
        return False
    total = max(1, counts.get("_rows", 0))
    cur = counts.setdefault("_unknown", {}).get(key, 0)
    cur_rate = cur / total
    if cur_rate >= target:
        return False
    margin = max(target - cur_rate, 0.0)
    base_p = min(0.5, target + margin)
    return random.random() < base_p

def _mark_unknown_inc(counts, key):
    counts.setdefault("_unknown", {})
    counts["_unknown"][key] = counts["_unknown"].get(key, 0) + 1

SEM = {
    "disease": {"organ": {}},
    "cell_type": {"organ": {}},
    "cell_line": {"organ": {}},
    "treatment": {"disease": {}},
    "treatment_time": {"treatment": {}},
    "response": {"treatment": {}},
    "biopsy_site": {"organ": {}},
    "library_source": {"cell_type": {}},
    "sex": {"organ": {}, "disease": {}}
}

DISEASE_TO_ORG = {
    "healthy": OPEN["organ"],
    "cholelithiasis": ["gallbladder","bile duct"],
    "acute cholecystitis": ["gallbladder"],
    "appendicitis": ["appendix"],
    "duodenal ulcer": ["duodenum"],
    "celiac disease": ["duodenum","jejunum"],
    "jejunitis": ["jejunum"],
    "infectious ileitis": ["ileum"],
    "laryngeal squamous cell carcinoma": ["larynx"],
    "laryngitis": ["larynx"],
    "tracheitis": ["trachea"],
    "urothelial carcinoma": ["bladder","ureter","urethra"],
    "acute cystitis": ["bladder"],
    "ureteral carcinoma": ["ureter"],
    "urethral carcinoma": ["urethra"],
    "endometrial carcinoma": ["endometrium","fallopian tube"],
    "mucoepidermoid carcinoma": ["salivary gland"],
    "tonsillitis": ["tonsil"],
    "pharyngitis": ["pharynx"],
    "spinal cord injury": ["spinal cord"],
    "meningitis": ["meninges"],
    "cholangiocarcinoma": ["bile duct"],
    "peritonitis": ["peritoneum"],
    "peritoneal mesothelioma": ["peritoneum","mesentery"],
    "fallopian tube carcinoma": ["fallopian tube"]
}
for d, organs in DISEASE_TO_ORG.items():
    SEM["disease"]["organ"][d] = [o for o in organs if o in OPEN["organ"]]

CELLTYPE_TO_ORG = {
    "goblet cells": ["duodenum","jejunum","ileum"],
    "paneth cells": ["duodenum","jejunum","ileum"],
    "enteric neurons": ["duodenum","jejunum","ileum"],
    "urothelial cells": ["bladder","ureter","urethra"],
    "umbrella cells": ["bladder"],
    "smooth muscle cells": ["bladder","ureter","fallopian tube","larynx","trachea"],
    "salivary acinar cells": ["salivary gland"],
    "taste receptor cells": ["tongue"],
    "laryngeal epithelial cells": ["larynx"],
    "tracheal ciliated cells": ["trachea"],
    "tonsillar lymphocytes": ["tonsil"],
    "mesothelial cells": ["peritoneum","mesentery"],
    "schwann cells": ["spinal cord"],
    "oligodendrocytes": ["spinal cord"],
    "ependymal cells": ["spinal cord"],
    "basal urothelial cells": ["bladder","ureter","urethra"],
    "interstitial cells of cajal": ["duodenum","jejunum","ileum"],
    "ductal epithelial cells": ["bile duct","salivary gland"],
    "follicular endometrial cells": ["endometrium"],
    "mucosal epithelial cells": ["pharynx","tonsil","larynx"],
    "epithelial cells": ["appendix","duodenum","jejunum","ileum","gallbladder","bile duct","larynx","trachea","salivary gland","tongue","bladder","ureter","urethra","endometrium","pharynx","tonsil","peritoneum","mesentery"]
}
for ct, organs in CELLTYPE_TO_ORG.items():
    SEM["cell_type"]["organ"][ct] = [o for o in organs if o in OPEN["organ"]]

CELLLINE_TO_ORG = {
    "GBC-SD": ["gallbladder"],
    "GBC-T": ["gallbladder"],
    "H69": ["bile duct"],
    "SCC-25": ["tongue"],
    "CAL 27": ["tongue"],
    "HSC-3": ["tongue"],
    "FaDu": ["pharynx"],
    "Detroit 562": ["pharynx"],
    "T24": ["bladder"],
    "RT4": ["bladder"],
    "UM-UC-3": ["bladder"],
    "HT-1376": ["bladder"],
    "5637": ["bladder"],
    "SNU-1076": ["larynx"],
    "Ben-Men-1": ["meninges"],
    "NCI-H2052": ["peritoneum"],
    "SNG-M": ["mesentery"],
    "HSG": ["salivary gland"],
    "OE-33D": ["duodenum"],
    "EM-E6/E7/hTERT": ["endometrium"]
}
for cl, organs in CELLLINE_TO_ORG.items():
    SEM["cell_line"]["organ"][cl] = [o for o in organs if o in OPEN["organ"]]

TREATMENT_TO_DISEASE = {
    "no treatment": ["healthy"],
    "cholecystectomy": ["cholelithiasis","acute cholecystitis"],
    "appendectomy": ["appendicitis"],
    "proton pump inhibitor therapy": ["duodenal ulcer"],
    "gluten-free diet": ["celiac disease"],
    "laryngectomy": ["laryngeal squamous cell carcinoma"],
    "tracheostomy": ["tracheitis","laryngeal squamous cell carcinoma"],
    "intravesical BCG": ["urothelial carcinoma"],
    "transurethral resection": ["urothelial carcinoma","urethral carcinoma"],
    "hysteroscopic curettage": ["endometrial carcinoma"],
    "salpingectomy": ["fallopian tube carcinoma"],
    "tonsillectomy": ["tonsillitis"],
    "broad-spectrum antibiotics": ["appendicitis","acute cystitis","peritonitis","meningitis","tracheitis","pharyngitis","laryngitis","infectious ileitis","jejunitis"],
    "spinal decompression surgery": ["spinal cord injury"],
    "meningitis antimicrobial therapy": ["meningitis"],
    "ERCP with stenting": ["cholangiocarcinoma","cholelithiasis"],
    "peritoneal lavage": ["peritonitis"],
    "platinum-based chemotherapy": ["urothelial carcinoma","peritoneal mesothelioma","laryngeal squamous cell carcinoma","cholangiocarcinoma","endometrial carcinoma","fallopian tube carcinoma","mucoepidermoid carcinoma"],
    "IMRT radiotherapy": ["laryngeal squamous cell carcinoma","urothelial carcinoma"],
    "mitomycin intravesical": ["urothelial carcinoma"],
    "endoscopic mucosal resection": ["duodenal ulcer"],
    "biliary drainage": ["cholangiocarcinoma","acute cholecystitis"],
    "ureteral stenting": ["ureteral carcinoma"],
    "urethroplasty": ["urethral carcinoma"],
    "hormonal therapy": ["endometrial carcinoma"],
    "unknown": OPEN["disease"]
}
for t, ds in TREATMENT_TO_DISEASE.items():
    SEM["treatment"]["disease"][t] = [d for d in ds if d in OPEN["disease"]]
for t in OPEN["treatment"]:
    if t not in SEM["treatment"]["disease"]:
        SEM["treatment"]["disease"][t] = []
for t in OPEN["treatment"]:
    SEM["treatment_time"]["treatment"][t] = OPEN["treatment_time"][:]
for t in OPEN["treatment"]:
    if t in {"gluten-free diet", "proton pump inhibitor therapy"}:
        SEM["response"]["treatment"][t] = ["stable","success","unknown"]
    elif any(k in t for k in ["chemotherapy","radiotherapy","platinum","mitomycin"]):
        SEM["response"]["treatment"][t] = ["stable","progressive","success","unknown"]
    elif t in {"no treatment","unknown"}:
        SEM["response"]["treatment"][t] = ["unknown"]
    else:
        SEM["response"]["treatment"][t] = ["success","unknown","stable"]
for site in OPEN["biopsy_site"]:
    if site in OPEN["organ"]:
        SEM["biopsy_site"]["organ"][site] = [site]
    elif site == "urine":
        SEM["biopsy_site"]["organ"][site] = ["bladder","ureter","urethra"]
    elif site == "bile":
        SEM["biopsy_site"]["organ"][site] = ["gallbladder","bile duct"]
for ct in OPEN["cell_type"]:
    SEM["library_source"]["cell_type"].setdefault("single-cell", [])
    SEM["library_source"]["cell_type"].setdefault("bulk", [])
    SEM["library_source"]["cell_type"].setdefault("spatial", [])
    prefs = []
    if any(w in ct for w in ["cells","cell","lymphocytes","neurons","schwann","oligodendrocytes","ependymal"]):
        prefs.append("single-cell")
    if any(w in ct for w in ["epithelial","acinar","mesothelial","ciliated","ductal","mucosal","follicular","urothelial"]):
        prefs.append("spatial")
    prefs.append("bulk")
    for src in set(prefs):
        if ct not in SEM["library_source"]["cell_type"][src]:
            SEM["library_source"]["cell_type"][src].append(ct)
SEM["sex"]["organ"]["fallopian tube"] = ["female"]
SEM["sex"]["organ"]["endometrium"] = ["female"]

def pick_balanced(counts, key, allowed):
    allowed = [v for v in allowed if v is not None]
    if not allowed:
        return None
    counts.setdefault(key, {})
    best = min([counts[key].get(v,0) for v in allowed])
    cands = [v for v in allowed if counts[key].get(v,0) == best]
    choice = random.choice(cands)
    counts[key][choice] = counts[key].get(choice,0)+1
    return choice

def choose_coherent_record(counts):
    d = pick_balanced(counts, "disease", OPEN["disease"])
    orgs = SEM["disease"]["organ"].get(d, OPEN["organ"])
    org = pick_balanced(counts, "organ", orgs if orgs else OPEN["organ"])
    ct_allowed = [ct for ct, os in SEM["cell_type"]["organ"].items() if org in os] or OPEN["cell_type"]
    ct = pick_balanced(counts, "cell_type", ct_allowed)
    cl_allowed = [cl for cl, os in SEM["cell_line"]["organ"].items() if org in os] or OPEN["cell_line"]
    cl = pick_balanced(counts, "cell_line", cl_allowed)
    src_pref = []
    for s, cts in SEM["library_source"]["cell_type"].items():
        if ct in cts: src_pref.append(s)
    src = pick_balanced(counts, "sequencing_source", src_pref if src_pref else CLOSED["sequencing_source"])
    if src in {"single-cell","spatial"}:
        lib_allowed = [v for v in CLOSED["library_selection"] if v not in {"inverse rRNA","hybrid selection"}]
    else:
        lib_allowed = CLOSED["library_selection"]
    lib = pick_balanced(counts, "library_selection", lib_allowed)
    is_cancer = "true" if ("carcinoma" in d or "mesothelioma" in d) else "false"
    bt_pool = CLOSED["biopsy_type"]
    if is_cancer == "false":
        bt_pool = ["primary","blood"]
    bt = pick_balanced(counts, "biopsy_type", bt_pool)
    if bt == "blood":
        if org in {"bladder","ureter","urethra"}:
            bs = "urine"
        elif org in {"gallbladder","bile duct"}:
            bs = "bile"
        else:
            bt = "primary"
            prim = SEM["biopsy_site"]["organ"].get(org, [org])
            prim = [x for x in prim if x in OPEN["biopsy_site"]] or [org]
            bs = pick_balanced(counts, "biopsy_site", prim)
    elif bt == "metastasis":
        alt = [x for x in OPEN["biopsy_site"] if x != org and x in OPEN["organ"]]
        bs = pick_balanced(counts, "biopsy_site", alt if alt else OPEN["biopsy_site"])
    else:
        prim = SEM["biopsy_site"]["organ"].get(org, [org])
        prim = [x for x in prim if x in OPEN["biopsy_site"]] or [org]
        bs = pick_balanced(counts, "biopsy_site", prim)
    if d == "healthy":
        t = "no treatment"
        tt = "unknown"
        rr = "unknown"
    else:
        t_allowed = [t for t, ds in SEM["treatment"]["disease"].items() if d in ds] or OPEN["treatment"]
        t = pick_balanced(counts, "treatment", t_allowed)
        if t in {"unknown","no treatment"}:
            tt = "unknown"
            rr = "unknown"
        else:
            tt = pick_balanced(counts, "treatment_time", SEM["treatment_time"]["treatment"].get(t, OPEN["treatment_time"]))
            rr = pick_balanced(counts, "response", SEM["response"]["treatment"].get(t, CLOSED["response"]))
    sex_pool = CLOSED["sex"]
    if org in SEM["sex"]["organ"]:
        sex_pool = SEM["sex"]["organ"][org]
    sex = pick_balanced(counts, "sex", sex_pool)
    age = pick_balanced(counts, "age", OPEN["age"])
    eth = pick_balanced(counts, "ethnicity", OPEN["ethnicity"])
    rec = {
        "library_selection": lib,
        "sequencing_source": src,
        "biopsy_site": bs,
        "biopsy_type": bt,
        "cell_line": cl,
        "cell_type": ct,
        "organ": org,
        "disease": d,
        "treatment": t,
        "treatment_time": tt,
        "response": rr,
        "age": age,
        "sex": sex,
        "ethnicity": eth,
        "is_cancer": is_cancer
    }
    rec = apply_unknowns(rec, counts)
    if rec["is_cancer"] == "false" and rec["biopsy_type"] == "metastasis":
        rec["biopsy_type"] = "primary"
        prim = SEM["biopsy_site"]["organ"].get(rec["organ"], [rec["organ"]])
        prim = [x for x in prim if x in OPEN["biopsy_site"]] or [rec["organ"]]
        rec["biopsy_site"] = random.choice(prim)
    return rec

def apply_unknowns(rec, counts):
    r = dict(rec)
    force_unknown_pairs = False
    if r["disease"] == "healthy":
        r["treatment"] = "no treatment"
        r["treatment_time"] = "unknown"
        r["response"] = "unknown"
        force_unknown_pairs = True
    if r.get("treatment") in {"unknown","no treatment"}:
        r["treatment_time"] = "unknown"
        r["response"] = "unknown"
        force_unknown_pairs = True
    for k in ["library_selection","sequencing_source","biopsy_site","biopsy_type","cell_line","cell_type","organ","disease","treatment","treatment_time","response","age","sex","ethnicity"]:
        if k == "is_cancer":
            continue
        if force_unknown_pairs and k in {"treatment_time","response"}:
            continue
        if k in {"treatment_time","response"} and r.get("treatment") in {"unknown","no treatment"}:
            continue
        if _should_set_unknown(k, counts):
            if k == "treatment":
                r["treatment"] = "unknown"
                r["treatment_time"] = "unknown"
                r["response"] = "unknown"
                _mark_unknown_inc(counts, "treatment")
                _mark_unknown_inc(counts, "treatment_time")
                _mark_unknown_inc(counts, "response")
            elif k == "disease":
                if random.random() < 0.5:
                    r["disease"] = "healthy"
                    r["treatment"] = "no treatment"
                    r["treatment_time"] = "unknown"
                    r["response"] = "unknown"
                else:
                    r["disease"] = "unknown"
                _mark_unknown_inc(counts, "disease")
            else:
                r[k] = "unknown"
                _mark_unknown_inc(counts, k)
    if r["is_cancer"] == "false" and r.get("biopsy_type") == "metastasis":
        r["biopsy_type"] = "primary"
        prim = SEM["biopsy_site"]["organ"].get(r["organ"], [r["organ"]])
        prim = [x for x in prim if x in OPEN["biopsy_site"]] or [r["organ"]]
        r["biopsy_site"] = random.choice(prim)
    return r

def _syn_list_for(cat, val):
    if not val:
        return []
    if isinstance(val, str) and val.lower() in {"unknown","no treatment","not applicable"}:
        return []
    lst = SYN.get(val)
    if lst:
        return lst
    man = MANUAL_SYNONYMS.get(cat, {}).get(val, [])
    if man:
        return ensure_minimum_synonyms(val, man, minimum=5)
    return ensure_minimum_synonyms(val, [], minimum=5)

def _norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())

def surface_for(cat, val):
    if not val:
        return ""
    if str(val).lower() in {"unknown","not applicable","no treatment"}:
        return ""
    syns = _syn_list_for(cat, val)
    return random.choice(syns) if syns else ""

def inject_value(text, ins):
    if not text.strip():
        return ins
    if not ins:
        return text
    toks = text.split()
    idx = random.randint(0, len(toks))
    return " ".join(toks[:idx]+[ins]+toks[idx:])

UPPER_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Z]{2,}[A-Z0-9_]*|[A-Z]+_[A-Z0-9_]+)(?![A-Za-z0-9_])"
)

def detect_token_category(tok):
    t = tok.replace("_","")
    if "TREATMENTTIME" in t or t.endswith("TIME"):
        return "treatment_time"
    keys = ["library_selection","sequencing_source","biopsy_site","biopsy_type","cell_line","cell_type","organ","disease","treatment","treatment_time","response","age","sex","ethnicity"]
    for k in sorted(keys, key=len, reverse=True):
        if k.replace("_","").upper() in t:
            return k
    return None

def replace_tokens(text, cat2surf):
    used = set()
    def repl(m):
        tok = m.group(0)
        cat = detect_token_category(tok)
        if cat is None:
            return tok
        rep = cat2surf.get(cat, None)
        if rep is None:
            return tok
        used.add(cat)
        return rep
    t = UPPER_TOKEN_RE.sub(repl, text)
    return t, used

def cleanup(x):
    x = re.sub(r"\s{2,}"," ",x).strip().strip(",").strip()
    x = re.sub(r"\s+([,.;:!?])", r"\1", x)
    return x

def value_in_phrase(phrase, val, cat=None):
    if not val or str(val).lower() in {"unknown","no treatment","not applicable"}:
        return True
    p = _norm(phrase)
    for s in _syn_list_for(cat, val) + [val]:
        ss = _norm(s)
        if ss and ss in p:
            return True
    return False

def ensure_phrase_coverage(phrase, rec):
    needed = ["library_selection","sequencing_source","biopsy_site","biopsy_type","cell_line","cell_type","organ","disease","treatment","treatment_time","response","age","sex","ethnicity"]
    skip_if = {"treatment_time","response"}
    out = phrase
    for k in needed:
        if k in skip_if and rec.get("treatment") in {"unknown","no treatment"}:
            continue
        v = rec.get(k,"")
        if not v or str(v).lower() in {"unknown","no treatment","not applicable"}:
            continue
        if not value_in_phrase(out, v, cat=k):
            s = surface_for(k, v) or v
            out = inject_value(out, s)
    return cleanup(out)

medical_noise_pool = [
    "Comprehensive chart review included longitudinal vital trends.",
    "Extended discussion of assay calibration, spike-in controls, and cross-batch normalization was documented.",
    "Reports referenced heterogeneous enhancement without definite diffusion restriction on follow-up.",
    "Clinical timeline described intermittent febrile episodes with negative and normal lactate.",
    "Quality control indicated RIN assessment, adapter trimming, and duplicate removal prior to quantification.",
    "Sequencing metrics detailed read depth distribution, UMI collision estimates, and ambient RNA modeling.",
    "Laboratory panel included CBC with differential, CRP kinetics, and comprehensive metabolic profiling."
]

def maybe_add_noise(text):
    if random.random() < 0.35:
        n = random.randint(1, 3)
        snippets = random.sample(medical_noise_pool, n)
        if random.random() < 0.5:
            return cleanup(text + " " + " ".join(snippets))
        else:
            return cleanup(" ".join(snippets) + " " + text)
    return text

def load_templates_json(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    out = []
    for it in data:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict) and "template" in it:
            out.append(it["template"])
        elif isinstance(it, (list,tuple)) and it:
            out.append(str(it[0]))
    return [x for x in out if isinstance(x,str) and x.strip()]

def load_templates_csv(path):
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")
    except Exception:
        df = pd.read_csv(path, dtype=str).fillna("")
    cols = [c for c in df.columns if isinstance(c,str)]
    texts = []
    for _, r in df.iterrows():
        s = " ".join([str(r[c]) for c in cols if str(r[c]).strip()])
        if s.strip():
            texts.append(s)
    return texts

def build_phrase_A(tpl, rec):
    t = tpl
    cats_ctx = ["organ","biopsy_site","disease","cell_type","cell_line","treatment","treatment_time","response"]
    for c in cats_ctx:
        s = surface_for(c, rec.get(c,""))
        if s and random.random() < 0.85:
            t = inject_value(t, s)
    extra = ["ethnicity","age","sex","sequencing_source","library_selection"]
    random.shuffle(extra)
    for c in extra:
        s = surface_for(c, rec.get(c,""))
        if s and random.random() < 0.6:
            t = inject_value(t, s)
    t = ensure_phrase_coverage(cleanup(t), rec)
    t = maybe_add_noise(t)
    t = ensure_phrase_coverage(cleanup(t), rec)
    return t

def build_phrase_B(tpl, rec):
    cat2surf = {c: surface_for(c, rec.get(c,"")) for c in rec}
    if rec.get("treatment") in {"unknown","no treatment"}:
        cat2surf["treatment_time"] = ""
        cat2surf["response"] = ""
    if rec.get("treatment_time"):
        tpl = re.sub(r'(?<![A-Za-z0-9])TREATMENT\s*[_-]?\s*TIME(?![A-Za-z0-9])', cat2surf.get("treatment_time",""), tpl)
        tpl = re.sub(r'(?<![A-Za-z0-9])_?TIME(?![A-Za-z0-9])', cat2surf.get("treatment_time",""), tpl)
    out, used = replace_tokens(tpl, cat2surf)
    for c in rec:
        s = cat2surf.get(c,"")
        if c not in used and s and random.random() < 0.7:
            out = inject_value(out, s)
    out = ensure_phrase_coverage(cleanup(out), rec)
    out = maybe_add_noise(out)
    out = ensure_phrase_coverage(cleanup(out), rec)
    if not out.strip():
        pool = [cat2surf.get(k,"") for k in ["organ","disease","biopsy_site","biopsy_type"] if cat2surf.get(k)]
        out = " ".join([p for p in pool if p]) if pool else ""
    return cleanup(out)

def next_run_id(i):
    return f"RUNA{str(i+1).zfill(7)}"

def generate_A(n, templates):
    counts = {}
    counts["_rows"] = 0
    rows = []
    if not templates:
        templates = ["Sample prepared for RNA analysis.","Transcriptome profiling performed.","Study metadata mentions clinical context."]
    for i in range(n):
        rec = choose_coherent_record(counts)
        counts["_rows"] += 1
        tpl = random.choice(templates)
        phrase = build_phrase_A(tpl, rec)
        rows.append({"run_accession": next_run_id(i), **rec, "phrase": phrase})
    return pd.DataFrame(rows)

def generate_B(n, templates):
    counts = {}
    counts["_rows"] = 0
    rows = []
    if not templates:
        templates = ["LIBRARY_SELECTION SEQUENCING_SOURCE ORGAN DISEASE TREATMENT TREATMENT_TIME RESPONSE AGE SEX ETHNICITY BIOPSY_SITE BIOPSY_TYPE CELL_LINE CELL_TYPE"]
    for i in range(n):
        rec = choose_coherent_record(counts)
        counts["_rows"] += 1
        tpl = random.choice(templates)
        phrase = build_phrase_B(tpl, rec)
        rows.append({"run_accession": next_run_id(i), **rec, "phrase": phrase})
    return pd.DataFrame(rows)

def ensure_dirs(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def main():
    tpls_a = load_templates_json(TEMPLATE_JSON_A)
    tpls_b = load_templates_csv(TEMPLATE_CSV_B)
    dfA = generate_A(1000, tpls_a)
    dfB = generate_B(1000, tpls_b)
    ensure_dirs(OUT_A); ensure_dirs(OUT_B)
    cols = ["run_accession","library_selection","sequencing_source","biopsy_site","biopsy_type","cell_line","cell_type","organ","disease","treatment","treatment_time","response","age","sex","ethnicity","is_cancer","phrase"]
    dfA = dfA[cols]
    dfB = dfB[cols]
    dfA.to_csv(OUT_A, index=False)
    dfB.to_csv(OUT_B, index=False)
    meta = {
        "CLOSED": CLOSED,
        "OPEN": OPEN,
        "UNK_PCT": UNK_PCT,
        "SYNONYMS": SYN,
        "SEM": SEM
    }
    with open("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/all_cat_sets/create_synt_data/meta_used.json","w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

