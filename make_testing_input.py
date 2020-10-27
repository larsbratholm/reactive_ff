import numpy as np
import glob

# parse atom types
# select structure subset
# 4 isomers of 5 structures
# parse coordinates
# create molpro files


def parse_atomtypes():
    molecule_size = {}
    filenames = glob.glob("./xyz/*_0.xyz")
    for i, filename in enumerate(filenames):
        if i % 10000 == 0:
            print(f"{i} / {len(filenames)} ")
        basename = filename.split("/")[-1].split("_")[0]
        atomtypes = []
        with open(filename) as f:
            for j, line in enumerate(f):
                if j == 0:
                    n = int(line.strip())
                    continue
                if j == 1:
                    continue
                tokens = line.split()
                atomtypes.append(tokens[0])
        types, count = np.unique(atomtypes, return_counts=True)
        identifier = ""
        c_index = np.where(types == "C")[0]
        h_index = np.where(types == "H")[0]
        n_index = np.where(types == "N")[0]
        o_index = np.where(types == "O")[0]
        for index in c_index, h_index, n_index, o_index:
            if len(index) > 0:
                c = str(count[index[0]])
                identifier += c
            else:
                identifier += "0"
            if index is not o_index:
                identifier += ","

        if n not in molecule_size:
            molecule_size[n] = {}
        if identifier not in molecule_size[n]:
            molecule_size[n][identifier] = []
        molecule_size[n][identifier].append(basename)

    return molecule_size

def get_structure_subset(molecule_db):
    structures = []
    n = 13
    keys = list(molecule_db[n].keys())
    ids = np.random.choice(keys, size=5, replace=False)
    for i in ids:
        basenames = molecule_db[n][i]
        mols = np.random.choice(basenames, size=4, replace=False)
        for m in mols:
            conf = np.random.randint(31)
            structures.append(f"./xyz/{m}_{conf}.xyz")
    return structures

def generate_input_files(structures):
    header = "geomtyp=xyz\n" + \
                    "nosym\n" + \
                    "noorient\n" + \
                    "geometry={\n"

    speed_test_structure = \
        "C       2.682553142765      -1.439438488647      -0.661046844118\n" + \
        "C       1.943855644586      -0.231765356275      -0.093682523204\n" + \
        "C       0.214003469184       0.800627472803       0.703680408037\n" + \
        "C      -0.272761558326      -0.082655788263       1.716016419494\n" + \
        "C      -0.783026666974       1.019409382396      -0.183699731572\n" + \
        "C      -2.437479436159       1.639341869019       0.547052656472\n" + \
        "C      -1.095843845065      -0.587960645040      -1.347071314975\n" + \
        "H       2.668871645538      -3.017723564592      -1.164885544584\n" + \
        "H       3.688844315079      -1.056204110642      -1.589804766081\n" + \
        "H       1.708153655008      -1.403899383845      -1.577207273869\n" + \
        "H       1.747952994639      -0.377012812560       1.281012291903\n" + \
        "H       2.138182319334       0.217769833242      -1.272152822352\n" + \
        "H       0.267994238462      -0.243965199797      -0.972520432611\n" + \
        "H       0.147259620911      -1.583197887363       1.448767398382\n" + \
        "H      -0.544480461986       0.439143127216       2.822159196983\n" + \
        "H      -1.852931601343       0.238259806771       1.810667160475\n" + \
        "H      -1.233712084174       1.383762226753      -1.491973354623\n" + \
        "H      -3.877201474844       1.925908017288       0.125626163703\n" + \
        "H      -2.976558319304       1.095695877634       1.546747540189\n" + \
        "H      -2.075916852998       2.840743486441       0.399549121191\n" + \
        "H      -1.718655188426      -0.759872452733      -2.441159919966\n" + \
        "H      -0.314343654904      -1.300766424846      -1.886688127806\n" + \
        "H      -2.333583469738      -1.375180840608      -1.110818580992"

    basis1 = "basis=avtz\n"
    basis2 = "basis=avqz\n"
    f12_basis1 = "basis=vdz-f12\n"
    f12_basis2 = "basis=vtz-f12\n"
    pno_f12_basis1 = "basis={\n" + \
                 "default=vdz-f12\n" + \
                 "set,jkfit,context=jkfit\n" + \
                 "default,avtz\n" + \
                 "set,mp2fit,context=mp2fit\n" + \
                 "default,avdz\n" + \
                 "set,ri,context=jkfit\n" + \
                 "default,avtz\n" + \
                 "}\n" + \
                 "explicit,ri_basis=ri,df_basis=mp2fit,df_basis_exch=jkfit\n"
    pno_basis1 = "basis={\n" + \
                 "default=avdz\n" + \
                 "set,jkfit,context=jkfit\n" + \
                 "default,avtz\n" + \
                 "set,mp2fit,context=mp2fit\n" + \
                 "default,avdz\n" + \
                 "set,ri,context=jkfit\n" + \
                 "default,avtz\n" + \
                 "}\n" + \
                 "explicit,ri_basis=ri,df_basis=mp2fit,df_basis_exch=jkfit\n"

    reference = "hf\nuccsd(t)-f12c,cabs_singles=1,scale_trip=1"
    method1 = "df-hf\ndf-ccsd"
    method2 = "df-hf\ndf-ccsd(t)"
    f12_method1 = "df-hf\ndf-ccsd-f12,cabs_singles=1"
    f12_method2 = "df-hf\ndf-ccsd(t)-f12,cabs_singles=1,scale_trip=1"
    local_method1 = "df-hf\ndf-luccsd"
    local_method2 = "df-hf\ndf-luccsd(t)"
    local_method3 = "df-hf\ndf-ldcsd"
    pno_method1a = "df-hf,basis=jkfit\npno-lccsd"
    pno_method1b = "df-hf\npno-lccsd"
    pno_method2a = "df-hf,basis=jkfit\npno-lccsd(t)"
    pno_method2b = "df-hf\npno-lccsd(t)"
    pno_f12_method1a = "df-hf,basis=jkfit\npno-lccsd-f12,cabs_singles=1"
    pno_f12_method1b = "df-hf\npno-lccsd-f12,cabs_singles=1"
    pno_f12_method2a = "df-hf,basis=jkfit\npno-lccsd(t)-f12,cabs_singles=1,scale_trip=1"
    pno_f12_method2b = "df-hf\npno-lccsd(t)-f12,cabs_singles=1,scale_trip=1"

    footers = []
    footers.append("\n}\n" + f12_basis2 + reference)
    footers.append("\n}\n" + basis1 + method1)
    footers.append("\n}\n" + basis2 + method1)
    footers.append("\n}\n" + basis1 + method2)
    footers.append("\n}\n" + basis2 + method2)
    footers.append("\n}\n" + f12_basis1 + f12_method1)
    footers.append("\n}\n" + f12_basis2 + f12_method1)
    footers.append("\n}\n" + f12_basis1 + f12_method2)
    footers.append("\n}\n" + f12_basis2 + f12_method2)
    footers.append("\n}\n" + basis1 + "local,thrbp=0.985\n" + local_method1 )
    footers.append("\n}\n" + basis2 + "local,thrbp=0.99\n" + local_method1 )
    footers.append("\n}\n" + basis1 + "local,thrbp=0.985\n" + local_method2 )
    footers.append("\n}\n" + basis2 + "local,thrbp=0.99\n" + local_method2 )
    footers.append("\n}\n" + basis1 + "local,thrbp=0.985\n" + local_method3 )
    footers.append("\n}\n" + basis2 + "local,thrbp=0.99\n" + local_method3 )
    footers.append("\n}\n" + pno_basis1 + pno_method1a)
    footers.append("\n}\n" + pno_basis1 + pno_method1a + ",domopt=tight")
    footers.append("\n}\n" + basis1 + pno_method1b)
    footers.append("\n}\n" + basis1 + pno_method1b + ",domopt=tight")
    footers.append("\n}\n" + pno_basis1 + pno_method2a)
    footers.append("\n}\n" + pno_basis1 + pno_method2a + ",domopt=tight")
    footers.append("\n}\n" + basis1 + pno_method2b)
    footers.append("\n}\n" + basis1 + pno_method2b + ",domopt=tight")
    footers.append("\n}\n" + pno_f12_basis1 + pno_f12_method1a)
    footers.append("\n}\n" + pno_f12_basis1 + pno_f12_method1a + ",domopt=tight")
    footers.append("\n}\n" + f12_basis2 + pno_f12_method1b)
    footers.append("\n}\n" + f12_basis2 + pno_f12_method1b + ",domopt=tight")
    footers.append("\n}\n" + pno_f12_basis1 + pno_f12_method2a)
    footers.append("\n}\n" + pno_f12_basis1 + pno_f12_method2a + ",domopt=tight")
    footers.append("\n}\n" + f12_basis2 + pno_f12_method2b)
    footers.append("\n}\n" + f12_basis2 + pno_f12_method2b + ",domopt=tight")

    for i, structure in enumerate(structures):
        with open(structure) as f:
            lines = f.readlines()
            coordinates = "".join(lines[2:])
        for j, footer in enumerate(footers):
            string = header + coordinates + footer
            with open(f"./inp/{j}_{i}.inp", "w") as f:
                f.write(string)

    for j, footer in enumerate(footers):
        string = header + speed_test_structure + footer
        with open(f"./inp/test_{j}.inp", "w") as f:
            f.write(string)

if __name__ == "__main__":
    molecule_db = parse_atomtypes()
    structures = get_structure_subset(molecule_db)
    generate_input_files(structures)



