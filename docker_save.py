# --coding:utf-8--
import os
import subprocess
import zipfile
import sys



def dockerSave(image_name,image_tag,output_folder):
    
    image_path = os.path.join(output_folder,f'{image_name}-{image_tag}.tar')
    # 停止 Docker 容器
    try:
        subprocess.run(['docker', 'stop', f'{image_name}:{image_tag}'], check=True)
    except:
        print("no docker to stop")
    subprocess.run(['docker', 'save', f'{image_name}:{image_tag}', '-o', image_path], check=True)
    print(f'Saved Docker image {image_name}:{image_tag} as {image_path}')
    # 压缩 tar 文件成 zip 文件
    zip_path = os.path.join(output_folder,f'{image_name}-{image_tag}.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(image_path)

    # 删除 tar 文件
    os.remove(image_path)
    print(f'Saved Docker image {image_name}:{image_tag} as {zip_path}')


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python docker_save.py <docker_image_name> <docker_image_tag> <docker_save_folder>")
        sys.exit(1)

    # 指定 Docker 镜像的名称和标签
    image_name = sys.argv[1]
    image_tag = sys.argv[2]
    # 保存 Docker 镜像到 tar 文件中
    output_folder = sys.argv[3]
    
    dockerSave(image_name, image_tag, output_folder)