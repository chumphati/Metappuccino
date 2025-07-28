import pandas as pd

valid_lineages = {
    "MCF7", "U87", "HEK293", "HeLa", "HCT116", "HEK293T", "H9", "HepG2", "SH-SY5Y", "293FT", "K562", "HeLa cell",
    "LC2/ad", "HFF-1", "293T", "HaCaT", "KMS-12-PE", "Calu3", "Huh-7", "MDA-MB-231", "Panc1", "SW48", "HeLa cells",
    "HCET", "HK-2", "A375", "Akata", "AsPC-1", "MM.1S", "HAP1", "KG1a", "A375 cell", "HEL9217", "U937", "143B", "HL60",
    "FaDu cells", "RWPE-1", "PEO4", "HAP1 cells", "Huh7", "NCI-H358", "GCIY", "KATO III", "BIONi010-C", "HT29",
    "MCF-10A", "SK-N-AS", "SHSY5Y", "22Rv1", "HUVEC", "OVCAR3", "WI38", "H2228", "LX-2", "HT1080", "IM95", "h-pES10",
    "MOLM13", "HUVEC cell", "IMR-90", "HK2", "Saos2", "MDA-MB231", "HUVEC/TERT2", "SUIT-2", "Hep3B", "AMO-1", "Reh",
    "GSU", "VCaP", "MV-4-11", "MG63", "H1299", "HCT-15", "HAP1 cell", "HEK293 cell", "KKU-213A", "MSTO-211H", "1231A3",
    "888-mel", "1BR3", "JHOC5", "SK-MEL-28 cell", "MCF 10A", "Met5A", "SiHa", "WAe001-A", "MDAMB231", "BxPC3", "KBM-7",
    "WM983B", "TIG108", "Capan-1", "SK-Hep1", "SW839", "Jurkat", "SKM1", "SKNO-1", "TE5", "A431", "MIAPaca-2", "786-O",
    "SNU423", "SNU475", "JHH7", "TE11", "RPE-hTERT", "CJ179", "Meg01 cell", "NT2/D1", "L3.6pl", "PANC1", "HOS",
    "TIG121", "PK-45P", "A498", "HepG2 cell", "MDA-MB-231 cell", "Mg63", "GC7", "IPC298", "COLO205", "PK-8", "COLO-320",
    "CHP-134", "NEC15", "NBsusSR", "ECC12", "IM95m", "SW-13", "D283 Med", "NCI-H82", "TE-1", "TE-5", "VMRC-RCW",
    "786-o", "HEK293F", "MT2", "697", "RCH", "SCMC-L1", "KOCL33", "KOCL50", "YACL95", "KOPN66bi", "HBL3", "HAL-O1",
    "Nalm6", "HeLa-S3", "173", "199", "273", "438", "537", "1317", "1376", "PEA2", "CAOV4", "OV-4485", "TOV-3133G",
    "OV-2085", "COV362", "IGROV1", "NIH:OVCAR3", "PEA1", "ES2", "U-251MG", "DLD1", "CLB-Ba", "HCT15", "PANC-1",
    "MiaPaca-2", "DAUDI", "BeWo", "HPB-ALL", "Mo", "MEG-01", "KYM-1", "Ki-JK", "Het-1A", "Caki1", "Caki2", "H929",
    "HS-SY-2", "Kasumi-1", "PC3", "HB1119", "RS4-11", "ML-2", "EOL-1", "Caki-1", "Ca9-22", "Daudi", "P2", "Kasumi2",
    "KOCL44", "KOCL45", "KOCL51", "KOCL58", "KOPB26", "KOPN1", "Kasumi8", "KOPN30bi", "Nalm1", "YAMN73", "Kasumi9",
    "L-ASK", "P30_OHK", "NAGL-1", "316", "443", "500", "601", "605", "1102", "1182", "1383", "WM1862", "T47D",
    "SKMEL30", "MELJUSO", "PEO16", "OV-90", "TOV-3041G", "FUOV1", "OVKATE", "OAW42", "TOV-1946", "OV-3133"
}

input_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_data.csv'
output_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_clean_cl_data.csv'

df = pd.read_csv(input_path)

def keep_row(output):
    if pd.isna(output):
        return True
    if "cell_line:" not in output:
        return True
    cell_line = output.split("cell_line:")[1].split()[0]
    return cell_line in valid_lineages or cell_line == "Primary"

df_filtered = df[df["output"].apply(keep_row)]

df_filtered.to_csv(output_path, index=False)