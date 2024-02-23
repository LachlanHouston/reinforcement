import fnmatch
import requests
from io import BytesIO
import zipfile
import os
import sys

print("Hello! This is an automatic updating script that will perform the following operations:")
print("1) Download the most current version of the course material from gitlab")
print("2) Check if you are missing any files and create them")
print("3) update this script to the most recent version")
print("4) Update certain files that you should not edit (_grade-scripts and so on) to the most recent version")

url_install = "https://02465material.pages.compute.dtu.dk/02465public/information/installation.html"
sdir = os.path.dirname(__file__)
dry = False

if "02465public" in sdir and "tuhe" in sdir:
    dry = True
    print("-"*100)
    print("It has been detected that this script is running on the teachers computer.")
    print("This means that your files will not be overwritten normally.")
    print("In the highly unusual case this is a mistake, please change dry=False in the code.")
    print("-"*100)
    # raise Exception("(teachers not to himself: Don't run this on your own computer)")


print("The script is being run using python version:", sys.executable)

if not os.path.basename(sdir) == "irlc":
    print("The script was unable to locate an 'irlc' folder. The most likely reason this occurs is that you have moved the location of the script, or that you have deleted the irlc folder. ")
    print("The current location of the script is:", sdir)
    print("Make sure this folder contains an irlc folder. If you have deleted it, simply start over with the installation instructions. ")
    sys.exit(1)  # Exit with error code 1

try:
    import unitgrade  # type: ignore
    # import irlc
except ImportError as e:
    print("Your python environment was unable to locate unitgrade")
    print("This means that you either did not install the software correctly, or that you installed it in the wrong python interpreter (i.e., you have multiple versions of python installed).")

    print("VS Code: Please select a different Python through the Command Palette (Ctrl+Shift+P) and choose ""Python: Select Interpreter"".")
    print("Try all the Pythons you can choose and run the script from them")
    print(f"See also {url_install}")
    sys.exit(1)  # Exit with error code 1

def read_and_extract_zip(url):
    # Download the zip file from the URL
    base_dir = url.split("/main/")[-1].split(".zip")[0]
    response = requests.get(url)
    local_students_folder = os.path.dirname(os.path.dirname(__file__))
    always_overwrite = ['irlc/update_files.py', 'irlc/__init__.py', 'irlc/tests/*', '**/unitgrade_data/*.pkl', 'irlc/car/*', 'irlc/gridworld/*', 'irlc/pacman/*', 'irlc/utils/*', '*_grade.py', '*/project*_tests.py']
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        zip_content = BytesIO(response.content)
        # Open the zip file using the zipfile module
        with zipfile.ZipFile(zip_content, 'r') as zip_ref:
            # List the files in the zip file
            # Iterate over the files in the zip file
            for file_name in zip_ref.filelist:
                # Read the content of each file
                if not file_name.is_dir():
                    rp = os.path.relpath(file_name.filename, base_dir)
                    new_path = os.path.join(local_students_folder, rp)
                    overwrite = [p for p in always_overwrite if fnmatch.fnmatch(rp, p)]
                    if len(overwrite) > 0 or not os.path.isfile(new_path):
                        commit = True
                        try:
                            if os.path.isfile(new_path):
                                with open(new_path, 'rb') as newf:
                                    if newf.read() == zip_ref.read(file_name.filename):
                                        commit = False
                                    else:
                                        commit = True
                        except Exception as e:
                            print("Problem reading local file", new_path)
                            pass

                        if commit:
                            print("> Overwriting...", new_path)
                            if not dry:
                                if not os.path.isdir(os.path.dirname(new_path)):
                                    os.makedirs(os.path.dirname(new_path))
                                with open(new_path, 'wb') as f:
                                    f.write(zip_ref.read(file_name.filename))
                    else:
                        pass
    else:
        print(f"Failed to download the zip file. Status code: {response.status_code}. The DTU Gitlab server may be overloaded, unavailable, or you have no network.")
    a = 34

# Replace 'your_zip_file_url' with the actual URL of the zip file
zip_file_url = 'https://gitlab.compute.dtu.dk/02465material/02465students/-/archive/main/02465students-main.zip'
read_and_extract_zip(zip_file_url)

try:
    import irlc
except ImportError as e:
    print("Oh no, Python encountered a problem during importing irlc.")
    import site
    print("")
    print("This is possibly because you moved or renamed the 02465students folder after the installation was completed, ")
    print("or because you selected another python interpreter than the one you used during install. ")
    print("Please move/rename the students folder back so it can be found at the this path again, and/or select another interpreter from the command pallette")
    print(f"See also {url_install}")
    sys.exit(1)  # Exit with error code 1

print("> The script terminated successfully. Your files should be up to date.")