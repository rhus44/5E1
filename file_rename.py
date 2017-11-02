import os

opendir = r'C:\MIT Training Images'

files=os.listdir(opendir);
x=10000
for filename in files:
    if filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".tif"):
        bsnm = os.path.splitext(filename)[1]
        nm = str(x) + bsnm
        print("Searching: "+nm)
        while nm in files:
            print("matchfound: "+nm)
            x = x + 1
            nm = str(x)+bsnm
        os.rename(os.path.join(opendir, filename), os.path.join(opendir, str(x) + bsnm))
        print("Changed "+filename+" to "+str(x)+bsnm)
        x = x + 1
