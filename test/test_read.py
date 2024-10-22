from shtoolkit.shread import read_technical_note, read_gia

slr_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\CSR_SLR_TN11E_TN11E.txt"
slr_file2 = "D:\\tvg_toolkit\\shtoolkit\\src\\shtoolkit\\data\\TN-14_C30_C20_GSFC_SLR.txt"
gsm_folder = "D:\\wjh_code\\TVG\\CSR\\unfilter"
gia_file1 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\ICE6G_D.txt"
gia_file2 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\Purcell16.txt"
gia_file3 = "D:\\tvg_toolkit\\tvg_toolkit\\data\\Caron18.txt"
deg1_file = "D:\\tvg_toolkit\\tvg_toolkit1\\data\\TN-13_GEOC_CSR_RL0602.txt"
file1 = [slr_file1, slr_file1]
file2 = [slr_file2, slr_file2]

a = read_technical_note.read_technical_note_c20_c30(slr_file2)
b = read_technical_note.read_technical_note_deg1(deg1_file)
c = read_gia.read_gia_model(gia_file3, 60, "C18")