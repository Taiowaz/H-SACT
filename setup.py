from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "sampler_core",
        ["src/utils/sampler_core.cpp"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    # 1. 项目元数据
    name="hsact",  # 建议使用一个能代表你整个项目的名字
    version="0.1.0",  # 项目版本号
    author="Albin",  # 你的名字或团队名 (替换 XXXX-2)
    author_email="taiowaz@gmail.com",  # 你的邮箱 (替换 XXXX-3)
    url="https://github.com/Taiowaz/GeoSTHN",  # 项目主页 (替换 XXXX-4)
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    setup_requires=["pybind11>=2.5"],
    python_requires=">=3.9",
)
