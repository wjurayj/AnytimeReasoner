from setuptools import setup, find_packages

setup(
    name='anytime_reasoner',
    version='0.0.0',
    description='Open-source training recipe for reproducing AnytimeReasoner.',
    author='Penghui Qi',
    author_email='qphutu@gmail.com',
    license='Apache License 2.0',
    url='https://github.com/sail-sg/AnytimeReasoner',
    packages=find_packages(include=['anytime_reasoner',]),
    install_requires=[
        'google-cloud-aiplatform',
        'latex2sympy2',
        'pylatexenc',
        'sentence_transformers',
        'tabulate',
        # 'flash_attn==2.7.3',
    ],
    python_requires='>=3.9',
)
