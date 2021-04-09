import os, pyexiv2

imageFileExt = ('.jpeg', '.jpg', '.png')

files = os.listdir('.')
filenames = filter(lambda fn: fn.endswith(imageFileExt), files)
filenames = list(filenames)

print(f"Warning: you are about to delete all metadata in these {len(files)} images: ")
for fn in filenames:
    print(fn)
print(f"Enter delete{len(files)} to continue: ", end = '')

respond = str(input())
if respond == f'delete{len(files)}':
    for fn in filenames:
        img = pyexiv2.Image(fn)
        img.clear_exif()
        img.clear_iptc()
        img.clear_xmp()
    print('metadata cleared')
else:
    print('abort')
