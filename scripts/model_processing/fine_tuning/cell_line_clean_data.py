import re

def load_cellosaurus_names(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = file.read()
    entries = re.findall(r'ID   (.+?)\n.*?AC   CVCL_[0-9]+.*?\n(?:SY   (.*?)\n)?', data, re.DOTALL)
    names = set()
    for primary, synonyms in entries:
        names.add(primary.strip().lower())
        if synonyms:
            for synonym in synonyms.split(';'):
                names.add(synonym.strip().lower())
    return names

def extract_names_from_text(text):
    return [re.sub(r'\s*\(.*?\)', '', part).strip() for part in text.split(';')]

def check_cell_lines(cell_lines, valid_names):
    result = {}
    for name in cell_lines:
        clean_name = name.strip().lower().replace(" cells", "").replace(" cell", "")
        result[name] = clean_name in valid_names
    return result

cellosaurus_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/data/raw/cellosaurus.txt'
valid_cell_lines = load_cellosaurus_names(cellosaurus_path)

raw_text = """Primary tissue (7966); MCF7 (907); A549 (808); DLD-1 (807); U87 (717); HEK293 (127); HeLa (105); HCT116 (97); HEK293T (61); H9 (56); MCF-7 (53); 201B7 (51); HepG2 (47); SH-SY5Y (32); U2OS (25); Nthy-ori 3-1 (25); RUES2 (24); RPE (22); 293FT (22); K562 (20); BJ (20); HeLa cell (19); LC2/ad (18); HFDPC (18); hiPSC P11025 (18); cell_line (18); HUDEP-2 (17); HFF-1 (17); 293T (16); HaCaT (15); KhES1 (15); TIG-3 (14); KMS-12-PE (14); HL-60 (14); Human SFTPC-GFP 201B7 iPSC reporter line (B2-3) (14); iPSC line (14); Calu3 (14); HUVEC cell line (14); A-549 (14); HeLa tet-off (13); cellline.GATA3.4 (13); THP-1 (13); HFF (12); Huh-7 (12); MDS-L (12); ff-iPSC (12); A549 lung adenocarcinomma cell (12); CC-LP-1 (12); 1BRhTERT (12); cellline.CDX2.3 (11); cellline.ESRRB.1 (11); MDA-MB-231 (11); Panc1 (11); SW48 (11); Hs 746.T (11); HMGU1 (11); HeLa cells (10); cellline.CDX2.4 (10); cellline.DMRT1.2 (10); HCET (10); HK-2 (10); MIA PaCa-2 cells (10); HT-29 (10); A375 (10); RC9 (10); MCF7 RB- (10); cellline.rtTA3G-PiggyBac.1 (9); cellline.GATA3.1 (9); cellline.Emerald.1 (9); cellline.Emerald.2 (9); cellline.rtTA3G.1 (9); Akata (9); iPSCs (9); Human umbilical vein endothelial cells (HUVECs) (9); AsPC-1 (9); LMSU (9); MM.1S (8); cellline.CDX2.1 (8); cellline.CDX2.6 (8); cellline.GATA3.5 (8); cellline.ASCL1.1 (8); cellline.CDX2.2 (8); cellline.SOX2.1 (8); HAP1 (8); TSct#1_CT27 (8); ES#1_SEES1 (8); KG1a (8); A375 cell (8); HEL9217 (8); NCI-N87 (8); U937 (8); MCF7 PacqR (8); MSC (7); 143B (7); HL60 (7); cellline.GATA3.202 (7); cellline.CDX2.5 (7); cellline.ASCL1.2 (7); cellline.ESRRB.2 (7); A549/hSLAM (7); SVts8 (7); SCRC-4000 (7); FaDu cells (7); MOLM-13 (7); HCC827 cell (7); RWPE-1 (7); PEO4 (7); primary cell line (7); HS-5 (7); MAN13 (7); cellline.SEES3.1 (6); cellline.GATA3.201 (6); cellline.DMRT1.1 (6); NHDF-Neo (6); HAP1 cells (6); MKCL (6); Huh7 (6); NCI-H2122 (6); NCI-H358 (6); NCI-N87 cell (6); MIA Paca-2 (6); GCIY (6); KATO III (6); A375, BRAF mutant-A375 isogenic cell line (6); pES10 (6); BIONi010-C (6); HT29 (6); MCF-10A (6); SK-N-AS (6); TIB-202 THP-1 + HeLa (5); Biken-HeLa cell (5); SHSY5Y (5); 22Rv1 (5); cellline.SOX2.2 (5); A549/Ctrl (5); HUVEC (5); T98 (5); SW480 (5); OVCAR3 (5); TL-Om1 (5); WI38 (5); ES#2_SEES4 (5); H2228 (5); ATN-1 (5); EM-E6/E7/TERT (5); LX-2 (5); HT1080 (5); hESC (5); IM95 (5); OCUM-1 (5); h-pES10 (5); CC3 (5); PC9 (5); MOLM13 (5); vCAF (human vulvar cancer-associated fibroblast cell) (5); HUVEC cell (5); IMR-90 (4); HK2 (4); Saos2 (4); cellline.MYOD1.1 (4); cellline.MEF2C.2 (4); cellline.GATA3.3 (4); cellline.FOXL2.2 (4); cellline.CDYL2.1 (4); cellline.TBX5.2 (4); cellline.FOSL1.1 (4); cellline.BCL6.2 (4); cellline.RUVBL2.1 (4); cellline.IRF1.2 (4); cellline.HNF4A.3 (4); cellline.HOXA9.2 (4); cellline.ID1.1 (4); cellline.ZNF706.1 (4); cellline.KLF9.2 (4); cellline.NRF1.1 (4); cellline.OVOL2.1 (4); cellline.FOSL2.2 (4); cellline.KLF4.1 (4); MDA-MB231 (4); TBX-4B T cells (4); Huh-7.5 (4); KC02-44D (4); HPS0001 201B7 (4); HUVEC/TERT2 (4); S2-CP8 (4); TCC-MESO-2 (4); MKN45 (4); ED cell (4); JET cell (4); SUIT-2 (4); Hep3B (4); F02-98 (4); AMO-1 (4); Reh (4); mTK6 (4); iPSC derived cell line (4); 501-mel (4); AGS (4); GSU (4); MKN7 (4); BC290321.2 (4); VCaP (4); cellline.RUVBL2.2 (4); KKU-100 (4); 253G1 (4); HuG1-N (4); MV-4-11 (4); T47D RB- (4); T47D CDK6H (4); MG63 (3); NC65 (3); H1299 (3); cellline.MYOD1.2 (3); cellline.JUN.1 (3); cellline.JUN.2 (3); cellline.FOXA1.1 (3); cellline.ESX1.1 (3); cellline.TBX5.1 (3); cellline.BCL6.1 (3); cellline.IRF1.1 (3); cellline.ZNF281.1 (3); cellline.NEUROG3.1 (3); cellline.ZIC1.2 (3); cellline.HNF4A.4 (3); cellline.ID1.2 (3); cellline.OTX1.1 (3); cellline.KLF3.1 (3); cellline.GRHL2.2 (3); cellline.PITX2.1 (3); cellline.MYBL2.1 (3); cellline.NFIC.2 (3); cellline.CUX1.2 (3); cellline.RUNX3.1 (3); cellline.RUNX3.2 (3); cellline.OVOL2.2 (3); cellline.CEBPB.1 (3); cellline.CEBPB.2 (3); cellline.DLX4.2 (3); cellline.HES1.2 (3); cellline.NEUROD2.2 (3); cellline.NEUROG2.2 (3); cellline.NANOG.1 (3); cellline.KLF4.2 (3); HCT-15 (3); HAP1 cell (3); HEK293 cell (3); TIG-3 cells (3); MMNK-1 (3); KKU-213A (3); AsPC-1 (human pancreatic adenocarcinoma cell line) (3); MSTO-211H (3); TSct#2_CT29 (3); wt_HTLV-1 infected clone3 (3); mut_3_HTLV-1 infected clone4 (3); wt_HTLV-1 infected clone1 (3); wt_HTLV-1 infected clone2 (3); mut_3_HTLV-1 infected clone1 (3); wt_HTLV-1 infected clone5 with CRISPR-Cas9 editing (3); SAS (3); 1231A3 (3); HSC-4 (3); 888-mel (3); Li-7 (3); immortalized human hepatocyte (3); 1BR3 (3); AT1BR (3); ATLD2 (3); HSC62 (3); JHOC5 (3); SAEC (3); SK-MEL-28 cell (3); MCF 10A (3); HPMC (3); Met5A (3); SiHa (3); mixed SOJD3, SEUR7, WT02C3 and WT02C9 (3); PB5 (3); GSS (3); HGC-27 (3); WAe001-A (3); OCMI_91s (3); mixed BIONi010-C and HMGU1 (3); MDAMB231 (3); HepaRG (3); G729 (3); BxPC3 (3); KBM-7 (3); cellline.TCF4.2 (3); cellline.FOXA1.2 (3); cellline.NFIC.1 (3); cellline.NFIB.1 (3); cellline.ERG.2 (3); wt_HTLV-1 infected clone5 (3); mut_3_HTLV-1 infected clone2 (3); Panc_1 (3); LCL (3); HCMEC (3); RERF-GC-1B (3); WM983B (3); peripheral blood mononuclear cells (2); RPTEC (2); TIG108 (2); Capan-1 (2); cellline.JUNB.1 (2); cellline.POU5F1.2 (2); cellline.MEF2C.1 (2); cellline.GATA3.2 (2); cellline.ESX1.2 (2); cellline.ZIC1.1 (2); cellline.HOXC9.1 (2); cellline.JUNB.4 (2); cellline.HSF1.1 (2); cellline.PITX2.2 (2); cellline.CUX1.1 (2); cellline.NRF1.2 (2); cellline.GLIS2.1 (2); cellline.GLIS2.2 (2); cellline.ERG.1 (2); cellline.GLI1.1 (2); cellline.NEUROD1.2 (2); cellline.NEUROD2.1 (2); cellline.NANOG.2 (2); cellline.SPIC.1 (2); cellline.SPIC.2 (2); cellline.GADD45A.2 (2); cellline.ZSCAN4.2 (2); cellline.SALL4.2 (2); cellline.ETS1.1 (2); cellline.KAT8.1 (2); cellline.KAT8.2 (2); cellline.IRF3.1 (2); cellline.FOXA2.1 (2); cellline.FOXA2.2 (2); cellline.E2F4.2 (2); cellline.SMAD7.1 (2); cellline.ETS2.1 (2); cellline.THAP11.1 (2); cellline.THAP11.2 (2); cellline.SOX9.2 (2); cellline.TFAP2A.1 (2); cellline.LHX2.1 (2); cellline.NELFA.1 (2); cellline.RXRA.1 (2); cellline.RXRA.2 (2); cellline.ATF1.1 (2); cellline.SMARCB1.2 (2); cellline.NKX2-1.1 (2); cellline.NKX2-1.2 (2); cellline.SPI1.1 (2); cellline.NEUROG1.1 (2); cellline.ARNT2.2 (2); cellline.ZNF274.1 (2); cellline.TFAP4.1 (2); cellline.NELFE.1 (2); cellline.SETDB1.2 (2); cellline.USF2.1 (2); cellline.OLIG2.2 (2); cellline.IRF4.1 (2); cellline.IRF4.2 (2); cellline.CTCFL.1 (2); cellline.ARID3A.2 (2); cellline.FOXP1.4 (2); cellline.TFE3.2 (2); cellline.E2F6.1 (2); cellline.STAT3.1 (2); cellline.HNF4A.1 (2); cellline.ATF3.2 (2); cellline.SREBF2.1 (2); cellline.SMARCA4.4 (2); cellline.GTF2F1.1 (2); cellline.GTF2F1.2 (2); cellline.PLXNB3.1 (2); cellline.PLXNB3.2 (2); cellline.RHOXF2.2 (2); cellline.RFX5.2 (2); cellline.STAT5A.1 (2); cellline.TGIF1.2 (2); cellline.ZBTB45.1 (2); cellline.TCF23.2 (2); cellline.EGFLAM.1 (2); cellline.EGFLAM.2 (2); cellline.FOXG1.1 (2); cellline.FOXG1.2 (2); cellline.IRF2.2 (2); cellline.ASCL2.2 (2); cellline.LMO1.1 (2); cellline.DLX3.1 (2); cellline.TFAP2C.2 (2); cellline.SOX15.2 (2); cellline.ZFAND3.2 (2); cellline.FOS.1 (2); cellline.FOS.2 (2); cellline.NFYB.1 (2); cellline.NFYB.2 (2); cellline.LMO2.1 (2); cellline.SMAD2.1 (2); cellline.TFCP2L1.1 (2); cellline.HHEX.1 (2); cellline.PTF1A.2 (2); cellline.STRA13.1 (2); cellline.TBX6.1 (2); cellline.TBX6.2 (2); cellline.SUB1.2 (2); cellline.SMAD5.2 (2); cellline.FBXO15.1 (2); cellline.MKRN1.1 (2); cellline.BRF2.2 (2); cellline.PBX1.1 (2); cellline.PBX1.2 (2); cellline.FOXM1.1 (2); cellline.HDAC8.1 (2); cellline.TSHZ1.1 (2); cellline.E2F1.1 (2); cellline.TRIM28.1 (2); cellline.MEIS1.3 (2); cellline.MEIS1.4 (2); cellline.ZBTB33.1 (2); cellline.SNAPC1.1 (2); cellline.SNAPC1.2 (2); cellline.VPS72.2 (2); cellline.CEBPA.1 (2); cellline.SSX6.2 (2); cellline.DEDD2.1 (2); cellline.DEDD2.2 (2); cellline.CEBPD.1 (2); cellline.FOXC1.1 (2); cellline.CBX2.1 (2); cellline.BACH1.2 (2); cellline.ANKRD22.1 (2); cellline.ANKRD22.2 (2); cellline.RARG.1 (2); cellline.RARG.2 (2); cellline.OSTF1.2 (2); cellline.NR3C1.1 (2); cellline.TEAD4.2 (2); cellline.PHF8.2 (2); cellline.GTF2B.1 (2); cellline.ESRRA.2 (2); cellline.RAD21.1 (2); cellline.MAX.2 (2); cellline.TGM2.1 (2); cellline.TGM2.2 (2); cellline.GTF3C2.2 (2); cellline.CTNNB1.2 (2); cellline.EED.1 (2); cellline.EED.2 (2); cellline.HMGB2.2 (2); cellline.HOXB4.2 (2); cellline.HDAC3.2 (2); cellline.EGR2.1 (2); cellline.TAL1.1 (2); cellline.TCEA3.1 (2); cellline.TCEA3.2 (2); cellline.MSC.1 (2); cellline.REST.1 (2); cellline.REST.2 (2); cellline.ISL2.1 (2); cellline.POU2AF1.1 (2); cellline.PHOX2B.1 (2); cellline.PHOX2B.2 (2); cellline.LHX1.1 (2); cellline.PAX5.2 (2); cellline.STAG1.1 (2); cellline.STAG1.2 (2); cellline.LHX3.1 (2); cellline.EN1.2 (2); cellline.BATF.1 (2); cellline.MAB21L2.1 (2); cellline.MAB21L2.2 (2); cellline.PCDH1.2 (2); cellline.HOPX.2 (2); cellline.CRY2.2 (2); cellline.STAT4.2 (2); cellline.ETV3.1 (2); cellline.ETV3.2 (2); cellline.HMGA2.2 (2); cellline.FOXP3.2 (2); cellline.TCF12.1 (2); cellline.KLF15.2 (2); cellline.PAX8.2 (2); cellline.CREB1.1 (2); cellline.CREB1.2 (2); cellline.SPZ1.1 (2); cellline.SPZ1.2 (2); cellline.RBPJ.1 (2); cellline.GABPA.2 (2); cellline.PKNOX2.1 (2); cellline.OLIG1.2 (2); cellline.POU3F2.1 (2); cellline.POU3F2.2 (2); cellline.SOX10.1 (2); cellline.ZNF713.2 (2); cellline.BLZF1.1 (2); cellline.BLZF1.2 (2); cellline.ZNF280B.1 (2); cellline.ZNF280B.2 (2); cellline.TCEB3.2 (2); cellline.POU5F1.3 (2); cellline.HMGXB4.1 (2); cellline.HMGXB4.2 (2); cellline.GLIS2.3 (2); cellline.NEUROD6.2 (2); cellline.WNT3A.1 (2); cellline.TFAP2B.2 (2); cellline.FEV.1 (2); cellline.FEV.2 (2); cellline.ZBTB3.2 (2); cellline.KLF5.2 (2); cellline.NKX6-2.1 (2); cellline.MSGN1.1 (2); cellline.MSGN1.2 (2); cellline.FOXD1.1 (2); cellline.NR4A2.2 (2); cellline.PITX3.2 (2); cellline.NKX2-2.1 (2); cellline.NKX2-2.2 (2); cellline.LHX4.1 (2); cellline.BANP.2 (2); cellline.TWIST2.2 (2); cellline.THAP3.1 (2); cellline.THAP3.2 (2); cellline.HOXD13.1 (2); cellline.HOXD13.2 (2); cellline.BARX2.1 (2); cellline.DLX5.1 (2); cellline.DLX5.2 (2); cellline.OTP.2 (2); cellline.POU4F3.1 (2); cellline.POU4F3.2 (2); cellline.SNAI2.1 (2); cellline.ALX3.1 (2); cellline.ALX3.2 (2); cellline.CREB3.2 (2); cellline.ETV1.1 (2); cellline.HOXD10.1 (2); cellline.HOXD10.2 (2); cellline.NFATC1.1 (2); cellline.DUX4.1 (2); cellline.ZIC3.1 (2); cellline.HOXB1.1 (2); cellline.HOXB1.2 (2); cellline.KLF12.1 (2); cellline.KLF11.1 (2); cellline.KLF11.2 (2); cellline.TBX4.2 (2); cellline.MYF5.2 (2); cellline.KLF8.2 (2); cellline.WT1.1 (2); cellline.BATF3.1 (2); cellline.NFE2L1.1 (2); cellline.NR2E1.1 (2); cellline.CCNE1.2 (2); cellline.NPAS2.2 (2); cellline.FLI1.4 (2); cellline.ISL1.1 (2); cellline.FOXD4L3.2 (2); cellline.NR1D2.1 (2); cellline.NR1D2.2 (2); cellline.NR4A3.1 (2); cellline.SOX5.1 (2); cellline.SIM1.2 (2); cellline.NR2C2.1 (2); cellline.NR2C2.2 (2); cellline.AR.1 (2); cellline.AR.2 (2); cellline.EVX1.1 (2); cellline.FOXD3.1 (2); cellline.DVL2.2 (2); cellline.HESX1.2 (2); cellline.KLF14.2 (2); cellline.TYRP1.1 (2); cellline.CDYL.2 (2); cellline.ZMYND12.1 (2); cellline.RBBP6.2 (2); cellline.LHX5.1 (2); cellline.CHD1.1 (2); cellline.CHD1.2 (2); cellline.E2F7.2 (2); cellline.IKZF1.2 (2); cellline.ETV6.1 (2); cellline.ETV6.2 (2); cellline.HMGN1.1 (2); SK-Hep1 (2); SW839 (2); Jurkat (2); primary (2); SKM1 (2); HepAD38.7 (2); SKNO-1 (2); KhES-1 (2); TE5 (2); PSC128 (human pancreatic stellate cell) (2); A431 (human vulvar epidermoid carcinoma cell line) (2); MIAPaca-2 (2); 786-O (2); WT10 (2); SNU423 (2); SNU475 (2); TSct#3_CT30 (2); TBX-4B cell (2); mut_3_HTLV-1 infected clone3 (2); Colorectal cancer cell line (2); periodontal ligament cells (2); JHH7 (2); GM18994 (2); GM18971 (2); GM19065 (2); GM19066 (2); TE11 (2); RPE-hTERT (2); CJ179 (2); Meg01 cell (2); NT2/D1 (2); GC10 (2); GC11 (2); GC14 (2); GC15 (2); GC16 (2); GC21 (2); GC8 (2); A549-ACE5 (2); A549-ACE6 (2); A549-ACE10 (2); CAF_OC229 (2); L3.6pl (2); Human Aortic Endothelial Cells (2); ACCX11 (2); PANC1 (2); INA-6 (2); TIB-202 THP-1 (2); HOS (2); TIG121 (2); PK-45P (2); cellline.FOSL1.2 (2); cellline.ZNF281.2 (2); cellline.ZNF706.2 (2); cellline.OTX1.2 (2); cellline.KLF9.1 (2); cellline.NFIB.2 (2); cellline.HES1.1 (2); cellline.GLI1.2 (2); cellline.FOSL2.1 (2); cellline.GADD45A.1 (2); cellline.ZSCAN4.1 (2); cellline.PDX1.2 (2); cellline.TBX3.1 (2); cellline.YY1.2 (2); cellline.SMAD7.2 (2); cellline.RARA.2 (2); cellline.SRSF6.2 (2); cellline.TFAP2A.2 (2); cellline.EGR1.1 (2); cellline.ARNT2.1 (2); cellline.IRF5.1 (2); cellline.SOX11.2 (2); cellline.NELFE.2 (2); cellline.CTCFL.2 (2); cellline.MEIS2.1 (2); cellline.MEIS2.2 (2); cellline.OTX2.2 (2); cellline.L3MBTL2.1 (2); cellline.RFX5.1 (2); cellline.HNF1A.1 (2); cellline.PPARG.1 (2); cellline.ETV5.2 (2); cellline.TRPV2.2 (2); cellline.DCP1A.1 (2); cellline.DCP1A.2 (2); cellline.RBBP5.2 (2); cellline.SMAD2.2 (2); cellline.MAB21L3.1 (2); cellline.XRCC4.2 (2); cellline.HDAC8.2 (2); cellline.TSHZ1.2 (2); cellline.GATA1.1 (2); cellline.VPS72.1 (2); cellline.BHLHE40.1 (2); cellline.CEBPA.2 (2); cellline.UGP2.1 (2); cellline.BARHL2.2 (2); cellline.HDAC2.1 (2); cellline.TEAD4.1 (2); cellline.TCF7L2.1 (2); cellline.ESRRA.1 (2); cellline.ESR1.1 (2); cellline.FIGLA.1 (2); cellline.EN1.1 (2); cellline.CRY2.1 (2); cellline.KLF15.1 (2); cellline.PAX2.1 (2); cellline.GATA6.2 (2); cellline.GABPA.1 (2); cellline.HOXD1.1 (2); cellline.HOXA3.2 (2); cellline.T.2 (2); cellline.SOX7.1 (2); cellline.NEUROD6.1 (2); cellline.POU4F1.1 (2); cellline.MSX2.1 (2); cellline.ZNF426.1 (2); cellline.NR4A2.1 (2); cellline.BARX2.2 (2); cellline.FOXB1.2 (2); cellline.SNAI2.2 (2); cellline.CTBP2.1 (2); cellline.POLR3A.1 (2); cellline.POLR3A.2 (2); cellline.NR2E1.2 (2); cellline.ELF4.4 (2); cellline.MEF2A.2 (2); cellline.FOXE3.1 (2); cellline.FOXD4L3.1 (2); cellline.FOXO3.1 (2); cellline.MIF4GD.1 (2); cellline.SOX6.1 (2); cellline.RBBP6.1 (2); cellline.CERS2.2 (2); cellline.IKZF1.1 (2); cellline.H2AFZ.2 (2); SKM-1 (2); A498 (2); HepG2 cell (2); MDSL (2); NHEK (2); Pancreatic cancer cell line (2); GM18965 (2); GM18978 (2); GM18979 (2); GM18985 (2); GM18987 (2); GM19077 (2); MDA-MB-231 cell (2); Mg63 (2); GC7 (2); A549-ACE7 (2); IPC298 (2); Huh7.5.1-8 (1); RPMI-8226 (1); COLO205 (1); CCSC#P (1); EBV-positive Akata (1); PK-8 (1); PK-45H (1); COLO-320 (1); CHP-134 (1); LK-2 (1); TC-YIK (1); NEC15 (1); NBsusSR (1); ECC12 (1); KHYG-1 (1); IM95m (1); G-401 (1); SW-13 (1); D283 Med (1); DMS 144 (1); NCI-H82 (1); TE-1 (1); TE-5 (1); cellline.HOXA9.1 (1); cellline.HOXC9.2 (1); cellline.JUNB.3 (1); cellline.MYBL2.2 (1); cellline.DLX4.1 (1); cellline.NEUROG2.1 (1); cellline.PDX1.1 (1); cellline.EOMES.1 (1); cellline.EOMES.2 (1); cellline.ELF5.2 (1); cellline.NR2F2.2 (1); cellline.ETS1.2 (1); cellline.IRF3.2 (1); cellline.E2F4.1 (1); cellline.SIX5.2 (1); cellline.TP53.1 (1); cellline.TP53.2 (1); cellline.ETS2.2 (1); cellline.SIN3A.1 (1); cellline.SIN3A.2 (1); cellline.SOX9.1 (1); cellline.SPI1.2 (1); cellline.PRDM1.2 (1); cellline.TP73.3 (1); cellline.EGR1.2 (1); cellline.IRF5.2 (1); cellline.ZNF274.2 (1); cellline.SOX11.1 (1); cellline.OLIG2.1 (1); cellline.E2F6.2 (1); cellline.CRY1.2 (1); cellline.STAT3.2 (1); cellline.GATA2.1 (1); cellline.GATA2.2 (1); cellline.HNF4A.2 (1); cellline.ZNF263.2 (1); cellline.ATF3.1 (1); cellline.L3MBTL2.2 (1); cellline.ZNF646.1 (1); cellline.RHOXF2.1 (1); cellline.TGIF1.1 (1); cellline.ELF1.1 (1); cellline.ELF1.2 (1); cellline.FOXP1.2 (1); cellline.RSPO1.2 (1); cellline.JAG1.2 (1); cellline.DLX3.2 (1); cellline.ZFAND3.1 (1); cellline.RBBP5.1 (1); cellline.FOXH1.1 (1); cellline.LMO2.2 (1); cellline.CTCF.2 (1); cellline.PAX6.1 (1); cellline.HHEX.2 (1); cellline.ELL2.1 (1); cellline.ELL2.2 (1); cellline.RFX2.1 (1); cellline.ATF2.1 (1); cellline.SUB1.1 (1); cellline.SUZ12.1 (1); cellline.HOXA2.1 (1); cellline.FBXO15.2 (1); cellline.TLK1.1 (1); cellline.FOXM1.2 (1); cellline.MEIS1.1 (1); cellline.LMX1A.2 (1); cellline.E2F1.2 (1); cellline.ZBTB33.2 (1); cellline.SSX6.1 (1); cellline.LIN28A.1 (1); cellline.SIX1.1 (1); cellline.FOXC1.2 (1); cellline.CBX2.2 (1); cellline.BACH1.1 (1); cellline.BARHL2.1 (1); cellline.PHF8.1 (1); cellline.ZKSCAN1.2 (1); cellline.GTF2B.2 (1); cellline.MAX.1 (1); cellline.REPIN1.1 (1); cellline.USF1.1 (1); cellline.VDR.2 (1); cellline.CTNNB1.1 (1); cellline.HDAC6.2 (1); cellline.TAL1.2 (1); cellline.RUNX1.2 (1); cellline.MSC.2 (1); cellline.STRA8.2 (1); cellline.YBX1.2 (1); cellline.DNMT3L.2 (1); cellline.HOXD3.1 (1); cellline.HOPX.1 (1); cellline.HOXA10.1 (1); cellline.HOXA10.2 (1); cellline.SMC3.4 (1); cellline.NR2F1.1 (1); cellline.MNX1.2 (1); cellline.TCF12.2 (1); cellline.ELK3.1 (1); cellline.SALL1.2 (1); cellline.PAX2.2 (1); cellline.THAP1.1 (1); cellline.ID3.1 (1); cellline.ID3.2 (1); cellline.FOXN3.2 (1); cellline.HOXD1.2 (1); cellline.PKNOX2.2 (1); cellline.SIRT3.1 (1); cellline.JARID2.1 (1); cellline.JUND.2 (1); cellline.SOX10.2 (1); cellline.ZNF713.1 (1); cellline.SNAI1.2 (1); cellline.PA2G4.1 (1); cellline.FOXS1.2 (1); cellline.NFIL3.2 (1); cellline.HOXB3.1 (1); cellline.KLF2.2 (1); cellline.TCEB3.1 (1); cellline.POU5F1.4 (1); cellline.HOXB4.3 (1); cellline.SOX7.2 (1); cellline.POU4F1.2 (1); cellline.TRAF4.1 (1); cellline.MYB.1 (1); cellline.SOX14.2 (1); cellline.RFXAP.1 (1); cellline.RFXAP.2 (1); cellline.KLF5.1 (1); cellline.NKX6-2.2 (1); cellline.SPIB.2 (1); cellline.PITX3.1 (1); cellline.MSI1.2 (1); cellline.BANP.1 (1); cellline.TLX3.1 (1); cellline.TLX3.2 (1); cellline.TWIST2.1 (1); cellline.GSC.1 (1); cellline.ETV2.2 (1); cellline.FOXB1.1 (1); cellline.OTP.1 (1); cellline.EHF.1 (1); cellline.EHF.2 (1); cellline.CREB3.1 (1); cellline.NFATC1.2 (1); cellline.FOXF1.1 (1); cellline.NR5A1.2 (1); cellline.ZIC3.2 (1); cellline.KLF12.2 (1); cellline.MYF5.1 (1); cellline.KLF8.1 (1); cellline.NEUROD4.1 (1); cellline.UBTF.1 (1); cellline.ELF5.4 (1); cellline.TP73.2 (1); cellline.NPAS2.1 (1); cellline.EEF1A1.1 (1); cellline.FOXE3.2 (1); cellline.ISL1.2 (1); cellline.FOXO3.2 (1); cellline.NR4A3.2 (1); cellline.SIM1.1 (1); cellline.NEUROD4.2 (1); cellline.FLI1.5 (1); cellline.CDKN1B.1 (1); cellline.CDKN1B.2 (1); cellline.FOXJ1.2 (1); cellline.EVX1.2 (1); cellline.ALX4.1 (1); cellline.ALX4.2 (1); cellline.POU4F1.3 (1); cellline.HESX1.1 (1); cellline.ZMYND12.2 (1); cellline.ZNF217.2 (1); cellline.CERS2.1 (1); cellline.LHX5.2 (1); cellline.MYT1.2 (1); cellline.ASH2L.2 (1); cellline.MAFK.1 (1); cellline.E2F7.1 (1); cellline.KDM1A.1 (1); cellline.KDM1A.2 (1); cellline.H2AFZ.1 (1); cellline.PABPC1.2 (1); RCC4 (1); ACHN (1); KMRC-1 (1); VMRC-RCW (1); TALL-1 (1); U-2 OS (1); 786-o (1); PT-5025 (1); 201-axon (1); 201-soma (1); 409-soma (1); 409-2-soma (1); CiRA26-axon (1); TDP10- axon (1); TDP10-soma (1); 201-B7-soma (1); A7-axon (1); D9-soma (1); AsPC-1, PSC128 (1); A431 (1); 787-O (1); wt_HTLV-1 infected clone4 (1); MT2-9-7 (1); SNT13 (1); SNT16 (1); SNT8 (1); HEK293F (1); GM18944 (1); GM18948 (1); GM19004 (1); GM18940 (1); GM18946 (1); GM18953 (1); GM18955 (1); GM18956 (1); GM18961 (1); GM18962 (1); GM18966 (1); GM18970 (1); GM18972 (1); GM18975 (1); GM18983 (1); GM18988 (1); GM18990 (1); GM18993 (1); GM19002 (1); GM19010 (1); GM19012 (1); GM19060 (1); GM19067 (1); GM19070 (1); GM19086 (1); GM19088 (1); MT2 (1); 697 (1); KOPN34 (1); KOPN36 (1); KOPN54 (1); RCH (1); SCMC-L1 (1); THP4 (1); YAMN92 (1); KOPN68 (1); KOCL33 (1); KOCL50 (1); YACL95 (1); KOPN66bi (1); HBL3 (1); KOPN46 (1); KOPN61 (1); KOPN62 (1); L-KUM (1); YAMN96 (1); YcuB4 (1); YcuB7 (1); HAL-O1 (1); YcuB2 (1); Nalm6 (1); HeLa-S3 (1); 66 (1); 71 (1); 99 (1); 104 (1); 149 (1); 167 (1); 170 (1); 172 (1); 173 (1); 178 (1); 179 (1); 199 (1); 273 (1); 298 (1); 368 (1); 379 (1); 406 (1); 408 (1); 435 (1); 436 (1); 438 (1); 441 (1); 442 (1); 461 (1); 465 (1); 475 (1); 478 (1); 482 (1); 499 (1); 504 (1); 507 (1); 508 (1); 509 (1); 533 (1); 535 (1); 537 (1); 557 (1); 578 (1); 581 (1); 583 (1); 584 (1); 585 (1); 589 (1); 602 (1); 659 (1); 718 (1); 737 (1); 751 (1); 779 (1); 849 (1); 850 (1); 887 (1); 889 (1); 914 (1); 927 (1); 964 (1); 977 (1); 1020 (1); 1047 (1); 1049 (1); 1056 (1); 1057 (1); 1059 (1); 1072 (1); 1075 (1); 1080 (1); 1085 (1); 1095 (1); 1105 (1); 1127 (1); 1156 (1); 1192 (1); 1237 (1); 1239 (1); 1242 (1); 1255 (1); 1273 (1); 1277 (1); 1313 (1); 1317 (1); 1321 (1); 1323 (1); 1328 (1); 1329 (1); 1330 (1); 1331 (1); 1345 (1); 1374 (1); 1376 (1); 1395 (1); 1407 (1); 1408 (1); 1409 (1); 1410 (1); 1414 (1); 1415 (1); 1416 (1); 1419 (1); 1424 (1); 1441 (1); 1446 (1); GC22 (1); NT8e (1); AW13516 (1); AW8507 (1); A549-ACE8 (1); TOV-2978G (1); PEO14 (1); PEA2 (1); CAOV4 (1); TOV-2835EP (1); CIOV1 (1); TOV-2881EP (1); COV644 (1); OV-4485 (1); TOV-3133G (1); CIOV2 (1); OV-2085(2) (1); COV362 (1); OV56 (1); TOV-3121D (1); OV-1369(2) (1); CAOV3 (1); PEO1 (1); IGROV1 (1); PEO23 (1); NA12878 (1); NIH:OVCAR3 (1); PEA1 (1); OV17R (1); ES2 (1); OV-1946 (1); primary tissue (1); U-251MG (1); MIA-PaCa2 (1); DLD1 (1); CLB-Ba (1); Huh7.5.1 (1); HCT15 (1); SW620 (1); CCSC#11 (1); PANC-1 (1); KLM-1 (1); MiaPaca-2 (1); HOPE (1); H-EMC-SS (1); DAUDI (1); BeWo (1); HPB-ALL (1); PCM6 (1); ECC4 (1); Mo (1); G-402 (1); MEG-01 (1); KYM-1 (1); Ki-JK (1); Het-1A (1); TE-8 (1); cellline.FOXL2.1 (1); cellline.CDYL2.2 (1); cellline.NEUROG3.2 (1); cellline.KLF3.2 (1); cellline.GRHL2.1 (1); cellline.HSF1.2 (1); cellline.FLI1.1 (1); cellline.FLI1.2 (1); cellline.NEUROD1.1 (1); cellline.SALL4.1 (1); cellline.MYC.1 (1); cellline.MYC.2 (1); cellline.TBX3.2 (1); cellline.YY1.1 (1); cellline.SIX5.1 (1); cellline.LHX2.2 (1); cellline.NELFA.2 (1); cellline.ATF1.2 (1); cellline.SMARCB1.1 (1); cellline.TFAP4.2 (1); cellline.SETDB1.1 (1); cellline.ARID3A.1 (1); cellline.FOXP1.1 (1); cellline.CRY1.1 (1); cellline.OTX2.1 (1); cellline.ZNF263.1 (1); cellline.NKX2-5.1 (1); cellline.NKX2-5.2 (1); cellline.MXI1.1 (1); cellline.SAP30.1 (1); cellline.SAP30.2 (1); cellline.RNF2.1 (1); cellline.RNF2.2 (1); cellline.ZNF646.2 (1); cellline.STAT5A.2 (1); cellline.KDM5A.1 (1); cellline.KDM5A.2 (1); cellline.ZBTB45.2 (1); cellline.TCF23.1 (1); cellline.FOXP1.3 (1); cellline.RSPO1.1 (1); cellline.DLX6.2 (1); cellline.ASCL2.1 (1); cellline.LMO1.2 (1); cellline.JAG1.1 (1); cellline.TBX2.1 (1); cellline.TBX2.2 (1); cellline.PPARG.2 (1); cellline.TFAP2C.1 (1); cellline.LHFP.1 (1); cellline.LHFP.2 (1); cellline.TRPV2.1 (1); cellline.FOXH1.2 (1); cellline.CTCF.1 (1); cellline.PAX6.2 (1); cellline.MAB21L3.2 (1); cellline.ZFP57.1 (1); cellline.ZFP57.2 (1); cellline.SMAD1.1 (1); cellline.SMAD1.2 (1); cellline.STRA13.2 (1); cellline.ATF2.2 (1); cellline.SUZ12.2 (1); cellline.SMAD5.1 (1); cellline.WRNIP1.1 (1); cellline.BRF2.1 (1); cellline.TLK1.2 (1); cellline.MEIS1.2 (1); cellline.CBX8.2 (1); cellline.TRIM28.2 (1); cellline.BHLHE40.2 (1); cellline.DLX2.1 (1); cellline.CEBPD.2 (1); cellline.SIX1.2 (1); cellline.UGP2.2 (1); cellline.ELK1.1 (1); cellline.ELK1.3 (1); cellline.HDAC2.2 (1); cellline.OSTF1.1 (1); cellline.NR3C1.3 (1); cellline.ESRRG.1 (1); cellline.ZKSCAN1.1 (1); cellline.CHD4.1 (1); cellline.CHD4.2 (1); cellline.RAD21.2 (1); cellline.WHSC1.1 (1); cellline.IRX2.1 (1); cellline.REPIN1.2 (1); cellline.GTF3C2.1 (1); cellline.VDR.1 (1); cellline.SIRT6.2 (1); cellline.ESR1.2 (1); cellline.HMGB2.1 (1); cellline.SMAD4.1 (1); cellline.SMAD4.2 (1); cellline.SNAI3.1 (1); cellline.VAX1.1 (1); cellline.VAX1.2 (1); cellline.NKX6-3.1 (1); cellline.YBX1.1 (1); cellline.POU2AF1.2 (1); cellline.LHX1.2 (1); cellline.FIGLA.2 (1); cellline.PAX5.1 (1); cellline.BATF.2 (1); cellline.DNMT3L.1 (1); cellline.PCDH1.1 (1); cellline.HOXD3.2 (1); cellline.HMGA2.1 (1); cellline.SMAD3.1 (1); cellline.SMAD3.2 (1); cellline.FOXP3.1 (1); cellline.PML.2 (1); cellline.NR2F1.2 (1); cellline.NKX6-1.2 (1); cellline.MNX1.1 (1); cellline.ELK3.2 (1); cellline.HIST2H3C.1 (1); cellline.HIST2H3C.2 (1); cellline.SALL1.1 (1); cellline.PAX8.1 (1); cellline.ATOH1.1 (1); cellline.ATOH1.2 (1); cellline.GBX2.2 (1); cellline.GBX2.4 (1); cellline.SPIC.3 (1); cellline.GATA6.1 (1); cellline.ZMAT4.1 (1); cellline.ZMAT4.2 (1); cellline.FOXN3.1 (1); cellline.RBPJ.2 (1); cellline.SIRT3.2 (1); cellline.JARID2.2 (1); cellline.JUND.1 (1); cellline.OLIG1.1 (1); cellline.EYA1.1 (1); cellline.AATF.2 (1); cellline.SNAI1.1 (1); cellline.HOXA3.1 (1); cellline.PA2G4.2 (1); cellline.NFIL3.1 (1); cellline.HOXB3.2 (1); cellline.KLF2.1 (1); cellline.T.1 (1); cellline.ELF5.3 (1); cellline.PATZ1.1 (1); cellline.NANOG.3 (1); cellline.SOX17.2 (1); cellline.THAP7.1 (1); cellline.TRAF4.2 (1); cellline.CCNE1.3 (1); cellline.MYB.2 (1); cellline.PPARGC1A.1 (1); cellline.PPARGC1A.2 (1); cellline.TFAP2B.1 (1); cellline.SOX14.1 (1); cellline.ZBED4.1 (1); cellline.ZBED4.3 (1); cellline.ZNF426.2 (1); cellline.SPIB.1 (1); cellline.FOXD1.2 (1); cellline.ELF2.2 (1); cellline.RREB1.1 (1); cellline.RREB1.2 (1); cellline.ETV2.1 (1); cellline.ETV1.2 (1); cellline.KLF1.2 (1); cellline.FOXF1.2 (1); cellline.NR5A1.1 (1); cellline.TBX4.1 (1); cellline.CTBP2.2 (1); cellline.EBF1.1 (1); cellline.WT1.2 (1); cellline.BATF3.2 (1); cellline.UBTF.2 (1); cellline.MYF6.1 (1); cellline.MYF6.2 (1); cellline.TAF7.2 (1); cellline.RUNX1.3 (1); cellline.EEF1A1.2 (1); cellline.ZBTB7A.1 (1); cellline.MEF2A.1 (1); cellline.MIF4GD.2 (1); cellline.DVL2.1 (1); cellline.BMP4.1 (1); cellline.ZNF217.1 (1); cellline.MYT1.1 (1); cellline.MAFK.2 (1); cellline.HMGN1.2 (1); cellline.PABPC1.1 (1); cellline.RPS6KA1.2 (1); A549/SeV-C (1); Caki1 (1); Caki2 (1); H929 (1); HS-SY-2 (1); Kasumi-1 (1); PC3 (1); 409-axon (1); 409-2-axon (1); CiRA26-soma (1); CiRA26-2-soma (1); D9-axon (1); F11-soma (1); A431, vCAF (1); HB1119 (1); MV4-11 (1); RS4-11 (1); ML-2 (1); EOL-1 (1); Caki-1 (1); NUGC3 (1); wt_HTLV-1 infected clone1 with CRISPR-Cas9 editing (1); Ca9-22 (1); Daudi (1); P3HR1 (1); GM18945 (1); GM18947 (1); GM18942 (1); GM18950 (1); GM18951 (1); GM18957 (1); GM18959 (1); GM18964 (1); GM18967 (1); GM18968 (1); GM18974 (1); GM18977 (1); GM18982 (1); GM18989 (1); GM18991 (1); GM18995 (1); GM19000 (1); GM19007 (1); GM19009 (1); GM19055 (1); GM19056 (1); GM19057 (1); GM19058 (1); GM19062 (1); GM19063 (1); GM19075 (1); GM19078 (1); GM19079 (1); GM19081 (1); GM19085 (1); P2 (1); Kasumi2 (1); KOPN63 (1); PreALP (1); YAMN90R (1); YcuB6 (1); YcuB8 (1); KOPN79 (1); KOCL44 (1); KOCL45 (1); KOCL51 (1); KOCL58 (1); KOPB26 (1); KOPN1 (1); THP8 (1); Kasumi8 (1); KCB1 (1); KOPN30bi (1); KOPN56 (1); KOPN83bi (1); Nalm1 (1); PALL-2 (1); SK9 (1); SU-Ph2 (1); TCCY (1); YAMN73 (1); Kasumi9 (1); L-ASK (1); MBMY (1); P30_OHK (1); THP5 (1); THP7 (1); YAMN74 (1); NAGL-1 (1); 8220 (1); 2102Ep (1); 18 (1); 46 (1); 57 (1); 67 (1); 69 (1); 72 (1); 74 (1); 81 (1); 82 (1); 88 (1); 89 (1); 94 (1); 98 (1); 107 (1); 169 (1); 180 (1); 181 (1); 215 (1); 216 (1); 235 (1); 256 (1); 283 (1); 295 (1); 306 (1); 310 (1); 316 (1); 330 (1); 366 (1); 369 (1); 380 (1); 392 (1); 400 (1); 413 (1); 422 (1); 423 (1); 437 (1); 443 (1); 471 (1); 479 (1); 480 (1); 500 (1); 506 (1); 546 (1); 558 (1); 560 (1); 587 (1); 597 (1); 598 (1); 601 (1); 605 (1); 639 (1); 687 (1); 756 (1); 803 (1); 805 (1); 811 (1); 818 (1); 842 (1); 876 (1); 896 (1); 912 (1); 922 (1); 926 (1); 963 (1); 969 (1); 971 (1); 981 (1); 984 (1); 1015 (1); 1026 (1); 1027 (1); 1030 (1); 1044 (1); 1048 (1); 1051 (1); 1079 (1); 1084 (1); 1087 (1); 1096 (1); 1102 (1); 1153 (1); 1158 (1); 1174 (1); 1180 (1); 1182 (1); 1194 (1); 1206 (1); 1236 (1); 1238 (1); 1243 (1); 1314 (1); 1315 (1); 1316 (1); 1319 (1); 1322 (1); 1332 (1); 1333 (1); 1334 (1); 1335 (1); 1372 (1); 1380 (1); 1383 (1); 1384 (1); 1394 (1); 1396 (1); 1398 (1); 1401 (1); 1411 (1); 1412 (1); 1443 (1); 1445 (1); GC1 (1); GC13 (1); GC23 (1); GC6 (1); OT9 (1); A549-ACE2 (1); A549-ACE3 (1); WM1862 (1); WM983A (1); T47D (1); SKMEL30 (1); MELJUSO (1); OV-866(2) (1); CIOV6 (1); 59M (1); PEO16 (1); PEO6 (1); OV-90 (1); FT194 (1); TOV-3041G (1); CIOV3 (1); OV-3331 (1); FUOV1 (1); TOV-3133D (1); OVKATE (1); COV362.4 (1); OVCAR-8 (1); OAW42 (1); TOV-1946 (1); OV-3133 (1)"""
cell_line_names = extract_names_from_text(raw_text)

results = check_cell_lines(cell_line_names, valid_cell_lines)

print("Valid lineage:")
for name, is_valid in results.items():
    if is_valid:
        print(name)

# Valid lineage:
# MCF7
# U87
# HEK293
# HeLa
# HCT116
# HEK293T
# H9
# HepG2
# SH-SY5Y
# 293FT
# K562
# HeLa cell
# LC2/ad
# HFF-1
# 293T
# HaCaT
# KMS-12-PE
# Calu3
# Huh-7
# MDA-MB-231
# Panc1
# SW48
# HeLa cells
# HCET
# HK-2
# A375
# Akata
# AsPC-1
# MM.1S
# HAP1
# KG1a
# A375 cell
# HEL9217
# U937
# 143B
# HL60
# FaDu cells
# RWPE-1
# PEO4
# HAP1 cells
# Huh7
# NCI-H358
# GCIY
# KATO III
# BIONi010-C
# HT29
# MCF-10A
# SK-N-AS
# SHSY5Y
# 22Rv1
# HUVEC
# OVCAR3
# WI38
# H2228
# LX-2
# HT1080
# IM95
# h-pES10
# MOLM13
# HUVEC cell
# IMR-90
# HK2
# Saos2
# MDA-MB231
# HUVEC/TERT2
# SUIT-2
# Hep3B
# AMO-1
# Reh
# GSU
# VCaP
# MV-4-11
# MG63
# H1299
# HCT-15
# HAP1 cell
# HEK293 cell
# KKU-213A
# MSTO-211H
# 1231A3
# 888-mel
# 1BR3
# JHOC5
# SK-MEL-28 cell
# MCF 10A
# Met5A
# SiHa
# WAe001-A
# MDAMB231
# BxPC3
# KBM-7
# WM983B
# TIG108
# Capan-1
# SK-Hep1
# SW839
# Jurkat
# SKM1
# SKNO-1
# TE5
# A431
# MIAPaca-2
# 786-O
# SNU423
# SNU475
# JHH7
# TE11
# RPE-hTERT
# CJ179
# Meg01 cell
# NT2/D1
# L3.6pl
# PANC1
# HOS
# TIG121
# PK-45P
# A498
# HepG2 cell
# MDA-MB-231 cell
# Mg63
# GC7
# IPC298
# COLO205
# PK-8
# COLO-320
# CHP-134
# NEC15
# NBsusSR
# ECC12
# IM95m
# SW-13
# D283 Med
# NCI-H82
# TE-1
# TE-5
# VMRC-RCW
# 786-o
# HEK293F
# MT2
# 697
# RCH
# SCMC-L1
# KOCL33
# KOCL50
# YACL95
# KOPN66bi
# HBL3
# HAL-O1
# Nalm6
# HeLa-S3
# 173
# 199
# 273
# 438
# 537
# 1317
# 1376
# PEA2
# CAOV4
# OV-4485
# TOV-3133G
# OV-2085
# COV362
# IGROV1
# NIH:OVCAR3
# PEA1
# ES2
# U-251MG
# DLD1
# CLB-Ba
# HCT15
# PANC-1
# MiaPaca-2
# DAUDI
# BeWo
# HPB-ALL
# Mo
# MEG-01
# KYM-1
# Ki-JK
# Het-1A
# Caki1
# Caki2
# H929
# HS-SY-2
# Kasumi-1
# PC3
# HB1119
# RS4-11
# ML-2
# EOL-1
# Caki-1
# Ca9-22
# Daudi
# P2
# Kasumi2
# KOCL44
# KOCL45
# KOCL51
# KOCL58
# KOPB26
# KOPN1
# Kasumi8
# KOPN30bi
# Nalm1
# YAMN73
# Kasumi9
# L-ASK
# P30_OHK
# NAGL-1
# 316
# 443
# 500
# 601
# 605
# 1102
# 1182
# 1383
# WM1862
# T47D
# SKMEL30
# MELJUSO
# PEO16
# OV-90
# TOV-3041G
# FUOV1
# OVKATE
# OAW42
# TOV-1946
# OV-3133
