import numpy as np
import glob

def write_input_files(structures, output_folder="./batch/"):
    header = "geomtyp=xyz\n" + \
                    "nosym\n" + \
                    "noorient\n" + \
                    "geometry={\n"

    #pno_f12_basis = "basis={\n" + \
    #             "default=vdz-f12\n" + \
    #             "set,jkfit,context=jkfit\n" + \
    #             "default,avtz\n" + \
    #             "set,mp2fit,context=mp2fit\n" + \
    #             "default,avdz\n" + \
    #             "set,ri,context=jkfit\n" + \
    #             "default,avtz\n" + \
    #             "}\n" + \
    #             "explicit,ri_basis=ri,df_basis=mp2fit,df_basis_exch=jkfit\n"

    #pno_f12_method = "df-hf,basis=jkfit\npno-lccsd(t)-f12,cabs_singles=1,scale_trip=1"

    #footer = "\n}\n" + pno_f12_basis + pno_f12_method + ",domopt=tight"

    footer = "\n}\nbasis=vtz-f12\ndf-hf\npno-lccsd(t)-f12,cabs_singles=1,scale_trip=1,domopt=tight"

    for i, structure in enumerate(structures):
        basename = structure.split("/")[-1].split(".")[0]
        with open(structure) as f:
            lines = f.readlines()
            coordinates = "".join(lines[2:])
        string = header + coordinates + footer
        with open(f"{output_folder}/{basename}.inp", "w") as f:
            f.write(string)

if __name__ == "__main__":
    filenames = glob.glob("./xyz/*.xyz")
    write_input_files(filenames)
