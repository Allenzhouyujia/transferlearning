#!/usr/bin/env python3
"""
🎓 Lesson 0: 环境搭建与配置验证

这是深度学习课程的第一课，我们将：
1. 验证Python和PyTorch环境
2. 检测和配置GPU设备
3. 测试基本功能
4. 生成环境报告
5. 为后续课程做准备

运行这个脚本来确保你的环境配置正确。
"""

import sys
import os
import platform
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class EnvironmentChecker:
    """环境检查器"""
    
    def __init__(self):
        self.results = {}
        self.recommendations = []
        
        # 颜色代码
        self.GREEN = '\033[92m'
        self.RED = '\033[91m'
        self.YELLOW = '\033[93m'
        self.BLUE = '\033[94m'
        self.BOLD = '\033[1m'
        self.END = '\033[0m'
    
    def print_colored(self, text: str, color: str):
        """打印彩色文本"""
        print(f"{color}{text}{self.END}")
    
    def print_header(self, title: str):
        """打印标题"""
        border = "=" * 60
        print(f"\n{self.BLUE}{border}")
        print(f"🎓 {title}")
        print(f"{border}{self.END}")
    
    def print_step(self, step: str):
        """打印步骤"""
        print(f"\n{self.BOLD}🔹 {step}{self.END}")
        print("-" * 40)
    
    def check_python_environment(self):
        """检查Python环境"""
        self.print_step("检查Python环境")
        
        # Python版本
        python_version = platform.python_version()
        version_tuple = tuple(map(int, python_version.split('.')))
        
        print(f"Python版本: {python_version}")
        print(f"Python执行路径: {sys.executable}")
        print(f"平台: {platform.platform()}")
        print(f"架构: {platform.machine()}")
        
        # 检查版本要求
        if version_tuple >= (3, 8):
            self.print_colored("✅ Python版本符合要求 (>= 3.8)", self.GREEN)
            python_ok = True
        else:
            self.print_colored("❌ Python版本过低，需要 >= 3.8", self.RED)
            self.recommendations.append("升级Python到3.8或更高版本")
            python_ok = False
        
        # 检查虚拟环境
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        if in_venv:
            self.print_colored("✅ 运行在虚拟环境中", self.GREEN)
        else:
            self.print_colored("⚠️  建议使用虚拟环境", self.YELLOW)
            self.recommendations.append("创建并激活虚拟环境")
        
        self.results['python'] = {
            'version': python_version,
            'version_ok': python_ok,
            'executable': sys.executable,
            'in_venv': in_venv,
            'platform': platform.platform()
        }
        
        return python_ok
    
    def check_pytorch_installation(self):
        """检查PyTorch安装"""
        self.print_step("检查PyTorch安装")
        
        try:
            import torch
            import torchvision
            
            torch_version = torch.__version__
            torchvision_version = torchvision.__version__
            
            print(f"PyTorch版本: {torch_version}")
            print(f"Torchvision版本: {torchvision_version}")
            
            # 检查CUDA支持
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                
                print(f"CUDA版本: {cuda_version}")
                print(f"GPU数量: {gpu_count}")
                print(f"主GPU: {gpu_name}")
                self.print_colored("✅ CUDA GPU支持可用", self.GREEN)
                
                device_info = {
                    'cuda_available': True,
                    'cuda_version': cuda_version,
                    'gpu_count': gpu_count,
                    'gpu_name': gpu_name
                }
            else:
                print("CUDA: 不可用")
                
                # 检查MPS (Apple Silicon)
                mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
                if mps_available:
                    self.print_colored("✅ MPS (Apple Silicon) 支持可用", self.GREEN)
                    device_info = {
                        'cuda_available': False,
                        'mps_available': True
                    }
                else:
                    self.print_colored("⚠️  将使用CPU，建议配置GPU加速", self.YELLOW)
                    self.recommendations.append("配置GPU支持以获得更好性能")
                    device_info = {
                        'cuda_available': False,
                        'mps_available': False
                    }
            
            self.print_colored("✅ PyTorch安装正常", self.GREEN)
            pytorch_ok = True
            
        except ImportError as e:
            self.print_colored(f"❌ PyTorch未安装或导入失败: {e}", self.RED)
            self.recommendations.append("安装PyTorch: pip install torch torchvision")
            pytorch_ok = False
            device_info = {}
        
        self.results['pytorch'] = {
            'installed': pytorch_ok,
            'version': torch_version if pytorch_ok else None,
            'torchvision_version': torchvision_version if pytorch_ok else None,
            **device_info
        }
        
        return pytorch_ok
    
    def check_additional_packages(self):
        """检查其他必需包"""
        self.print_step("检查其他依赖包")
        
        required_packages = {
            'matplotlib': '数据可视化',
            'numpy': '数值计算',
            'pandas': '数据处理',
            'seaborn': '统计可视化',
            'tqdm': '进度条',
            'PIL': '图像处理 (Pillow)',
            'sklearn': '机器学习工具'
        }
        
        package_status = {}
        all_packages_ok = True
        
        for package, description in required_packages.items():
            try:
                if package == 'PIL':
                    import PIL
                    version = PIL.__version__
                elif package == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                else:
                    module = __import__(package)
                    version = getattr(module, '__version__', 'Unknown')
                
                print(f"✅ {package}: {version} - {description}")
                package_status[package] = {'installed': True, 'version': version}
                
            except ImportError:
                print(f"❌ {package}: 未安装 - {description}")
                package_status[package] = {'installed': False, 'version': None}
                all_packages_ok = False
                self.recommendations.append(f"安装 {package}")
        
        if all_packages_ok:
            self.print_colored("✅ 所有依赖包都已安装", self.GREEN)
        else:
            self.print_colored("⚠️  部分依赖包缺失", self.YELLOW)
            self.recommendations.append("运行: pip install matplotlib numpy pandas seaborn tqdm pillow scikit-learn")
        
        self.results['packages'] = package_status
        return all_packages_ok
    
    def test_basic_functionality(self):
        """测试基本功能"""
        self.print_step("测试基本功能")
        
        try:
            import torch
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 测试PyTorch基本操作
            print("🧪 测试PyTorch张量操作...")
            x = torch.randn(3, 3)
            y = torch.mm(x, x)
            print(f"   张量形状: {x.shape} → {y.shape}")
            
            # 测试GPU/MPS（如果可用）
            if torch.cuda.is_available():
                print("🧪 测试CUDA操作...")
                x_gpu = x.cuda()
                y_gpu = torch.mm(x_gpu, x_gpu)
                print(f"   GPU张量计算成功: {y_gpu.device}")
                
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("🧪 测试MPS操作...")
                x_mps = x.to('mps')
                y_mps = torch.mm(x_mps, x_mps)
                print(f"   MPS张量计算成功: {y_mps.device}")
            
            # 测试数据可视化
            print("🧪 测试数据可视化...")
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
            ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
            ax.set_title("测试图表")
            plt.close(fig)  # 关闭图表避免显示
            print("   matplotlib绘图测试通过")
            
            # 测试数据处理
            print("🧪 测试数据处理...")
            arr = np.random.randn(100, 10)
            mean = np.mean(arr, axis=0)
            print(f"   NumPy数组处理成功: {arr.shape} → {mean.shape}")
            
            self.print_colored("✅ 所有功能测试通过", self.GREEN)
            functionality_ok = True
            
        except Exception as e:
            self.print_colored(f"❌ 功能测试失败: {e}", self.RED)
            self.recommendations.append("检查包安装和配置")
            functionality_ok = False
        
        self.results['functionality'] = {'basic_tests_passed': functionality_ok}
        return functionality_ok
    
    def test_cifar10_loading(self):
        """测试CIFAR-10数据加载"""
        self.print_step("测试CIFAR-10数据加载")
        
        try:
            import torchvision
            import torchvision.transforms as transforms
            
            print("📦 尝试加载CIFAR-10数据集...")
            
            # 创建数据目录
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            
            # 定义变换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            # 加载少量数据进行测试
            print("   正在下载/验证CIFAR-10数据集...")
            start_time = time.time()
            
            testset = torchvision.datasets.CIFAR10(
                root=data_dir, 
                train=False, 
                download=True, 
                transform=transform
            )
            
            end_time = time.time()
            
            print(f"   测试集大小: {len(testset)} 张图片")
            print(f"   数据加载耗时: {end_time - start_time:.1f} 秒")
            
            # 测试数据访问
            sample_image, sample_label = testset[0]
            print(f"   样本形状: {sample_image.shape}")
            print(f"   数据类型: {sample_image.dtype}")
            
            self.print_colored("✅ CIFAR-10数据集加载成功", self.GREEN)
            cifar10_ok = True
            
        except Exception as e:
            self.print_colored(f"❌ CIFAR-10数据加载失败: {e}", self.RED)
            self.recommendations.append("检查网络连接和存储空间")
            cifar10_ok = False
        
        self.results['cifar10'] = {'loading_successful': cifar10_ok}
        return cifar10_ok
    
    def performance_benchmark(self):
        """性能基准测试"""
        self.print_step("性能基准测试")
        
        try:
            import torch
            
            # 选择设备
            if torch.cuda.is_available():
                device = torch.device('cuda')
                device_name = torch.cuda.get_device_name(0)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                device_name = "Apple Silicon GPU"
            else:
                device = torch.device('cpu')
                device_name = "CPU"
            
            print(f"🚀 在 {device_name} 上进行性能测试...")
            
            # 矩阵乘法基准测试
            sizes = [500, 1000]
            results = {}
            
            for size in sizes:
                print(f"   测试 {size}x{size} 矩阵乘法...")
                
                # 创建随机矩阵
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                
                # 预热
                for _ in range(3):
                    _ = torch.mm(a, b)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # 性能测试
                start_time = time.time()
                for _ in range(5):
                    c = torch.mm(a, b)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 5
                
                # 计算性能指标
                ops = 2 * size ** 3  # 矩阵乘法的浮点运算数
                gflops = ops / (avg_time * 1e9)
                
                results[f"matrix_{size}"] = {
                    'time_ms': avg_time * 1000,
                    'gflops': gflops
                }
                
                print(f"     平均时间: {avg_time*1000:.2f} 毫秒")
                print(f"     性能: {gflops:.2f} GFLOPS")
            
            self.print_colored("✅ 性能基准测试完成", self.GREEN)
            
            self.results['performance'] = {
                'device': str(device),
                'device_name': device_name,
                'benchmarks': results
            }
            
            return True
            
        except Exception as e:
            self.print_colored(f"❌ 性能测试失败: {e}", self.RED)
            self.results['performance'] = {'error': str(e)}
            return False
    
    def generate_report(self):
        """生成环境报告"""
        self.print_step("生成环境报告")
        
        # 生成报告
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': {
                'python_version': platform.python_version(),
                'platform': platform.platform(),
                'machine': platform.machine()
            },
            'check_results': self.results,
            'recommendations': self.recommendations
        }
        
        # 保存到文件
        report_dir = project_root / "experiments"
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / "environment_report.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 报告已保存到: {report_file}")
        
        # 显示摘要
        print("\n📊 环境检查摘要:")
        print("-" * 30)
        
        checks = [
            ("Python环境", self.results.get('python', {}).get('version_ok', False)),
            ("PyTorch安装", self.results.get('pytorch', {}).get('installed', False)),
            ("依赖包", all(pkg['installed'] for pkg in self.results.get('packages', {}).values())),
            ("基本功能", self.results.get('functionality', {}).get('basic_tests_passed', False)),
            ("CIFAR-10数据", self.results.get('cifar10', {}).get('loading_successful', False))
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}")
            if not passed:
                all_passed = False
        
        print("\n" + "="*60)
        if all_passed:
            self.print_colored("🎉 环境配置完美！可以开始学习了！", self.GREEN)
            print("\n🚀 下一步：")
            print("   cd ../lesson1_dev_environment")
            print("   python main.py")
        else:
            self.print_colored("⚠️  环境配置需要完善", self.YELLOW)
            print("\n🔧 建议解决以下问题：")
            for rec in self.recommendations:
                print(f"   • {rec}")
        
        return all_passed
    
    def run_complete_check(self):
        """运行完整的环境检查"""
        self.print_header("Lesson 0: 环境配置验证")
        
        print("欢迎来到CIFAR-10深度学习课程！")
        print("让我们检查你的环境配置是否已经准备就绪...\n")
        
        # 运行所有检查
        checks = [
            self.check_python_environment,
            self.check_pytorch_installation,
            self.check_additional_packages,
            self.test_basic_functionality,
            self.test_cifar10_loading,
            self.performance_benchmark
        ]
        
        all_passed = True
        
        for check in checks:
            try:
                result = check()
                if not result:
                    all_passed = False
            except Exception as e:
                self.print_colored(f"❌ 检查过程中出现错误: {e}", self.RED)
                all_passed = False
        
        # 生成报告
        self.generate_report()
        
        return all_passed

def main():
    """主函数"""
    print("🎓 CIFAR-10深度学习课程")
    print("Lesson 0: 环境配置验证\n")
    
    checker = EnvironmentChecker()
    
    try:
        success = checker.run_complete_check()
        
        if success:
            print(f"\n{checker.GREEN}✅ 环境检查全部通过！{checker.END}")
            print("🎯 你已经准备好开始深度学习之旅了！")
            return 0
        else:
            print(f"\n{checker.YELLOW}⚠️  环境配置需要完善{checker.END}")
            print("🔧 请按照上面的建议完善环境配置")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n{checker.YELLOW}⚠️  检查被用户中断{checker.END}")
        return 1
    except Exception as e:
        print(f"\n{checker.RED}❌ 意外错误: {e}{checker.END}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 