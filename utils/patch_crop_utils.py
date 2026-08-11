"""
基于patch相似度比较的图像裁剪工具

功能：比较两张图片，根据颜色直方图相似度保留差异较大的patch，
     并根据连通性自动裁剪出有意义的矩形区域
"""

import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine
from scipy.ndimage import binary_erosion, binary_dilation
from collections import deque
from typing import List, Tuple, Optional, Dict


def compute_color_histogram(patch: Image.Image, bins: int = 16) -> np.ndarray:
    """
    计算单个patch的RGB颜色直方图
    
    Args:
        patch: PIL Image patch
        bins: 每个颜色通道的bin数量
    
    Returns:
        归一化的颜色直方图向量
    """
    patch_array = np.array(patch)
    
    # 计算每个通道的直方图
    hist_r = np.histogram(patch_array[:, :, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(patch_array[:, :, 1], bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(patch_array[:, :, 2], bins=bins, range=(0, 256))[0]
    
    # 连接三个通道的直方图
    hist = np.concatenate([hist_r, hist_g, hist_b])
    
    # 归一化
    hist = hist.astype(float)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist /= hist_sum
    
    return hist


def histogram_similarity(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    计算两个直方图的余弦相似度
    
    Returns:
        相似度分数，范围 [0, 1]，1表示完全相同
    """
    return 1 - cosine(hist1, hist2)


def find_connected_components(
    patch_mask: np.ndarray, 
    n_patches_h: int, 
    n_patches_w: int
) -> List[List[Tuple[int, int]]]:
    """
    找到所有连通的patch区域（使用BFS）
    
    Args:
        patch_mask: 2D numpy array，True表示保留的patch，False表示被mask的patch
        n_patches_h: patch高度数量
        n_patches_w: patch宽度数量
    
    Returns:
        每个连通区域包含的patch坐标列表
    """
    visited = np.zeros((n_patches_h, n_patches_w), dtype=bool)
    components = []
    
    # 四个方向：上、下、左、右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            if patch_mask[i, j] and not visited[i, j]:
                # BFS找到一个连通区域
                component = []
                queue = deque([(i, j)])
                visited[i, j] = True
                
                while queue:
                    ci, cj = queue.popleft()
                    component.append((ci, cj))
                    
                    # 检查四个方向的邻居
                    for di, dj in directions:
                        ni, nj = ci + di, cj + dj
                        if (0 <= ni < n_patches_h and 0 <= nj < n_patches_w and
                            patch_mask[ni, nj] and not visited[ni, nj]):
                            visited[ni, nj] = True
                            queue.append((ni, nj))
                
                components.append(component)
    
    return components


def get_bounding_box_from_patches(
    patches: List[Tuple[int, int]], 
    patch_size: int, 
    width: int, 
    height: int
) -> Optional[Tuple[int, int, int, int]]:
    """
    根据patch列表计算最小外接矩形
    
    Args:
        patches: patch坐标列表 [(i, j), ...]
        patch_size: patch大小
        width, height: 图像尺寸
    
    Returns:
        (left, top, right, bottom): 矩形边界，如果patches为空则返回None
    """
    if not patches:
        return None
    
    min_i = min(p[0] for p in patches)
    max_i = max(p[0] for p in patches)
    min_j = min(p[1] for p in patches)
    max_j = max(p[1] for p in patches)
    
    left = min_j * patch_size
    top = min_i * patch_size
    right = min((max_j + 1) * patch_size, width)
    bottom = min((max_i + 1) * patch_size, height)
    
    return (left, top, right, bottom)


def crop_image_by_patch_similarity(
    compared_img: Image.Image,
    base_img: Image.Image,
    patch_size: int = 28,
    similarity_threshold: float = 0.85,
    min_component_size: int = 10,
    dilation_iterations: int = 1,
    erosion_iterations: int = 1,
    verbose: bool = False
) -> List[Dict]:
    """
    基于patch相似度比较裁剪图像
    
    Args:
        compared_img: 要裁剪的图像
        base_img: 参考基准图像
        patch_size: patch大小（默认28，对应Qwen2VL的有效patch size）
        similarity_threshold: 相似度阈值，超过此值的patch会被忽略
        min_component_size: 最小连通区域大小，小于此值的区域会被过滤
        dilation_iterations: 形态学膨胀操作的迭代次数，用于连接邻近区域
        erosion_iterations: 形态学腐蚀操作的迭代次数，用于减少碎片化
        verbose: 是否打印详细信息
    
    Returns:
        裁剪结果列表，每个元素包含:
        {
            'image': PIL.Image - 裁剪后的图像,
            'bbox': (left, top, right, bottom) - 边界框,
            'patch_count': int - 包含的patch数量,
            'component_index': int - 连通区域索引
        }
    """
    # torch tensor to PIL Image
    

    # 确保输入是RGB模式
    if compared_img.mode != 'RGB':
        compared_img = compared_img.convert('RGB')
    if base_img.mode != 'RGB':
        base_img = base_img.convert('RGB')
    
    # 确保两张图片尺寸一致
    if compared_img.size != base_img.size:
        if verbose:
            print(f"Warning: Images have different sizes. Resizing base_img from {base_img.size} to {compared_img.size}")
        base_img = base_img.resize(compared_img.size, Image.LANCZOS)
    
    width, height = compared_img.size
    
    # 计算patch数量（包括边缘不完整的patch）
    n_patches_w = (width + patch_size - 1) // patch_size
    n_patches_h = (height + patch_size - 1) // patch_size
    
    if verbose:
        print(f"Image size: {width}x{height}")
        print(f"Patch size: {patch_size}x{patch_size}")
        print(f"Number of patches: {n_patches_w} x {n_patches_h} = {n_patches_w * n_patches_h}")
        print(f"Similarity threshold: {similarity_threshold}")
    
    # 创建patch mask矩阵：True表示保留，False表示被mask
    patch_mask = np.zeros((n_patches_h, n_patches_w), dtype=bool)
    
    # 统计变量
    total_patches = 0
    kept_patches = 0
    
    # 遍历所有patch
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # 计算patch的边界
            left = j * patch_size
            top = i * patch_size
            right = min(left + patch_size, width)
            bottom = min(top + patch_size, height)
            
            # 提取patch
            compared_patch = compared_img.crop((left, top, right, bottom))
            base_patch = base_img.crop((left, top, right, bottom))
            
            # 计算颜色直方图
            hist_compared = compute_color_histogram(compared_patch)
            hist_base = compute_color_histogram(base_patch)
            
            # 计算相似度
            similarity = histogram_similarity(hist_compared, hist_base)
            
            total_patches += 1
            
            # 如果相似度低于阈值，保留这个patch
            if similarity < similarity_threshold:
                patch_mask[i, j] = True
                kept_patches += 1
    
    if verbose:
        print(f"Kept patches: {kept_patches}/{total_patches} ({kept_patches/total_patches*100:.2f}%)")
    
    # 先进行形态学膨胀操作，连接邻近的区域
    if dilation_iterations > 0:
        if verbose:
            print(f"\nApplying morphological dilation (iterations: {dilation_iterations})...")
        before_dilation = np.sum(patch_mask)
        patch_mask = binary_dilation(patch_mask, iterations=dilation_iterations)
        after_dilation = np.sum(patch_mask)
        if verbose:
            print(f"Patches before dilation: {before_dilation}")
            print(f"Patches after dilation: {after_dilation}")
            print(f"Patches added by dilation: {after_dilation - before_dilation}")
    
    # 进行形态学腐蚀操作，以减少碎片化和连通区域分裂
    if erosion_iterations > 0:
        if verbose:
            print(f"\nApplying morphological erosion (iterations: {erosion_iterations})...")
        before_erosion = np.sum(patch_mask)
        patch_mask = binary_erosion(patch_mask, iterations=erosion_iterations)
        after_erosion = np.sum(patch_mask)
        if verbose:
            print(f"Patches before erosion: {before_erosion}")
            print(f"Patches after erosion: {after_erosion}")
            print(f"Patches removed by erosion: {before_erosion - after_erosion}")
    
    # 找到所有连通区域
    components = find_connected_components(patch_mask, n_patches_h, n_patches_w)
    
    if verbose:
        print(f"Found {len(components)} connected components")
    
    # 过滤掉过小的连通区域
    valid_components = [comp for comp in components if len(comp) >= min_component_size]
    
    if verbose:
        print(f"Valid components (size >= {min_component_size}): {len(valid_components)}")
    
    # 为每个有效连通区域裁剪图片
    cropped_results = []
    for idx, component in enumerate(valid_components):
        bbox = get_bounding_box_from_patches(component, patch_size, width, height)
        if bbox:
            left, top, right, bottom = bbox
            cropped = compared_img.crop(bbox)
            
            result = {
                'image': cropped,
                'bbox': bbox,
                'patch_count': len(component),
                'component_index': idx
            }
            
            if verbose:
                print(f"Component {idx}: {len(component)} patches, "
                      f"bbox=({left}, {top}, {right}, {bottom}), "
                      f"size={right-left}x{bottom-top}")
            
            cropped_results.append(result)
    
    return cropped_results


def crop_image_by_path(
    base_img_path: str,
    compared_img_path: str,
    patch_size: int = 28,
    similarity_threshold: float = 0.85,
    min_component_size: int = 10,
    dilation_iterations: int = 1,
    erosion_iterations: int = 1,
    output_dir: Optional[str] = None,
    verbose: bool = False
) -> List[Dict]:
    """
    从文件路径加载图像并进行裁剪
    
    Args:
        compared_img_path: 要裁剪的图像路径
        base_img_path: 参考基准图像路径
        patch_size: patch大小
        similarity_threshold: 相似度阈值
        min_component_size: 最小连通区域大小
        dilation_iterations: 形态学膨胀操作的迭代次数
        erosion_iterations: 形态学腐蚀操作的迭代次数
        output_dir: 输出目录（可选），如果提供则自动保存裁剪结果
        verbose: 是否打印详细信息
    
    Returns:
        裁剪结果列表
    """
    # 加载图像
    compared_img = Image.open(compared_img_path).convert('RGB')
    base_img = Image.open(base_img_path).convert('RGB')
    
    if verbose:
        print(f"Loaded compared image: {compared_img_path} ({compared_img.size})")
        print(f"Loaded base image: {base_img_path} ({base_img.size})")
    
    # 裁剪
    results = crop_image_by_patch_similarity(
        compared_img=compared_img,
        base_img=base_img,
        patch_size=patch_size,
        similarity_threshold=similarity_threshold,
        min_component_size=min_component_size,
        dilation_iterations=dilation_iterations,
        erosion_iterations=erosion_iterations,
        verbose=verbose
    )
    
    # 如果提供了输出目录，保存结果
    if output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for result in results:
            filename = f"crop_component_{result['component_index']}.png"
            output_path = os.path.join(output_dir, filename)
            result['image'].save(output_path)
            result['output_path'] = output_path
            
            if verbose:
                print(f"Saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    compared_path = "/apdcephfs_cq11/share_300483685/neoxlzhang/Sketch/Pair2/Raw/BigTrain/0/shot_0000/frame_02.jpg"
    base_path = "/apdcephfs_cq11/share_300483685/neoxlzhang/Sketch/Pair2/Raw/BigTrain/0/shot_0000/frame_03.jpg"
    output_dir = "./cropped_output"
    
    print("=" * 60)
    print("Patch-based Image Cropping Tool")
    print("=" * 60)
    
    results = crop_image_by_path(
        compared_img_path=compared_path,
        base_img_path=base_path,
        patch_size=28,
        similarity_threshold=0.85,
        min_component_size=10,
        dilation_iterations=1,
        erosion_iterations=1,
        output_dir=output_dir,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print(f"Cropping completed! Found {len(results)} valid regions.")
    print("=" * 60)
