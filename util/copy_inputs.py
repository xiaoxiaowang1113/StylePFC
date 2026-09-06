import os, shutil, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cnt', default='./data/cnt', help='Content image directory.')
parser.add_argument('--sty', default='./data/sty', help='Style image directory.')
opt = parser.parse_args()


def list_flat_pngs(folder_path):
    return [
        {
            'src_path': os.path.join(folder_path, filename),
            'name': os.path.splitext(filename)[0],
        }
        for filename in sorted(os.listdir(folder_path))
        if filename.endswith('.png')
    ]


def list_style_pngs(folder_path):
    style_images = []
    for root, _, files in os.walk(folder_path):
        png_files = sorted(filename for filename in files if filename.endswith('.png'))
        for filename in png_files:
            src_path = os.path.join(root, filename)
            rel_dir = os.path.relpath(root, folder_path)
            stem = os.path.splitext(filename)[0]
            if rel_dir == '.':
                style_name = stem
            else:
                style_name = f"{rel_dir.replace(os.sep, '_')}_{stem}"
            style_images.append(
                {
                    'src_path': src_path,
                    'name': style_name,
                }
            )
    return style_images


# inputs of directory
a_folder_path = opt.cnt
b_folder_path = opt.sty

# destination directory of copied inputs
result_folder_path = a_folder_path + '_eval'    # "./data/cnt_eval"
result_folder_path_ = b_folder_path + '_eval'   # "./data/sty_eval"

# get images in the directories
a_images = list_flat_pngs(a_folder_path)
b_images = list_style_pngs(b_folder_path)

if len(a_images) == 0:
    raise RuntimeError(f'No png content images found in: {a_folder_path}')
if len(b_images) == 0:
    raise RuntimeError(f'No png style images found in: {b_folder_path}')

# if no dst, make directory
if not os.path.exists(result_folder_path):
    os.makedirs(result_folder_path)
if not os.path.exists(result_folder_path_):
    os.makedirs(result_folder_path_)

# copy content and style images by all combination of pairs
for a_image in a_images:
    for b_image in b_images:
        result_filename = f"{a_image['name']}_stylized_{b_image['name']}.png"
        result_path = os.path.join(result_folder_path, result_filename)
        shutil.copy(a_image['src_path'], result_path)

for a_image in a_images:
    for b_image in b_images:
        result_filename = f"{a_image['name']}_stylized_{b_image['name']}.png"
        result_path = os.path.join(result_folder_path_, result_filename)
        shutil.copy(b_image['src_path'], result_path)

print(f'Content input: {a_folder_path}')
print(f'Style input: {b_folder_path}')
print(f'Content eval output: {result_folder_path}')
print(f'Style eval output: {result_folder_path_}')
print(f'Content images: {len(a_images)}')
print(f'Style images: {len(b_images)}')
print(f'Total pair copies per eval directory: {len(a_images) * len(b_images)}')
