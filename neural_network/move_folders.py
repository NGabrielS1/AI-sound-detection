import shutil
import os

fake_num = 0
real_num = 0

fake_dest = "data/fake"
real_dest = "data/real"

real_folders = ["data/training/real", "data/testing/real", "data/validation/real"]
fake_folders = ["data/training/fake", "data/testing/fake", "data/validation/fake"]

for folder in real_folders:
    for filename in os.listdir(folder):
        if os.path.splitext(filename)[1] == ".wav":
            shutil.move(os.path.join(folder,filename), os.path.join(real_dest,str(real_num)+".wav"))
            real_num += 1

for folder in fake_folders:
    for filename in os.listdir(folder):
        if os.path.splitext(filename)[1] == ".wav":
            shutil.move(os.path.join(folder,filename), os.path.join(fake_dest,str(fake_num)+".wav"))
            fake_num += 1