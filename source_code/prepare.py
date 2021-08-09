import os
import shutil

train_all_path = "/home/chisc/workspace/wuzhenrong"

train_dir = "/home/chisc/workspace/wuzhenrong/train"
validation_dir = "/home/chisc/workspace/wuzhenrong/validation"
test_dir = "/home/chisc/workspace/wuzhenrong/test"

train_cat = "/home/chisc/workspace/wuzhenrong/train/cat"
train_dog = "/home/chisc/workspace/wuzhenrong/train/dog"

val_cat = "/home/chisc/workspace/wuzhenrong/validation/cat"
val_dog = "/home/chisc/workspace/wuzhenrong/validation/dog"

test_cat = "/home/chisc/workspace/wuzhenrong/test/cat"
test_dog = "/home/chisc/workspace/wuzhenrong/test/dog"

if not os.path.exists(train_dir):
    os.mkdir(train_dir)
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

if not os.path.exists(train_cat):
    os.mkdir(train_cat)
if not os.path.exists(train_dog):
    os.mkdir(train_dog)

if not os.path.exists(val_cat):
    os.mkdir(val_cat)
if not os.path.exists(val_dog):
    os.mkdir(val_dog)

if not os.path.exists(test_cat):
    os.mkdir(test_cat)
if not os.path.exists(test_dog):
    os.mkdir(test_dog)

for i in range(0, 2000):
    addr = f"/home/chisc/workspace/wuzhenrong/train_all/cat.{i}.jpg"
    to_add = f"/home/chisc/workspace/wuzhenrong/train/cat/cat.{i}.jpg"
    shutil.copyfile(addr, to_add)
for i in range(0, 2000):
    addr = f"/home/chisc/workspace/wuzhenrong/train_all/dog.{i}.jpg"
    to_add = f"/home/chisc/workspace/wuzhenrong/train/dog/dog.{i}.jpg"
    shutil.copyfile(addr, to_add)


for i in range(2000, 2500):
    addr = f"/home/chisc/workspace/wuzhenrong/train_all/cat.{i}.jpg"
    to_add = f"/home/chisc/workspace/wuzhenrong/validation/cat/cat.{i}.jpg"
    shutil.copyfile(addr, to_add)
for i in range(2000, 2500):
    addr = f"/home/chisc/workspace/wuzhenrong/train_all/dog.{i}.jpg"
    to_add = f"/home/chisc/workspace/wuzhenrong/validation/dog/dog.{i}.jpg"
    shutil.copyfile(addr, to_add)


for i in range(2500, 3000):
    addr = f"/home/chisc/workspace/wuzhenrong/train_all/cat.{i}.jpg"
    to_add = f"/home/chisc/workspace/wuzhenrong/test/cat/cat.{i}.jpg"
    shutil.copyfile(addr, to_add)
for i in range(2500, 3000):
    addr = f"/home/chisc/workspace/wuzhenrong/train_all/dog.{i}.jpg"
    to_add = f"/home/chisc/workspace/wuzhenrong/test/cat/dog.{i}.jpg"
    shutil.copyfile(addr, to_add)