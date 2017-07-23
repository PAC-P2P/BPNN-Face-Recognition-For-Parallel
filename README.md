# 基于并行BP神经网络的人脸识别系统

此为 **并行版** 的人脸识别系统

并行版请戳：[Github . PAC-P2P/BPNN-Face-Recognition-For-Parallel](https://github.com/PAC-P2P/BPNN-Face-Recognition-For-Parallel)

串行版请戳：[Github . PAC-P2P/BPNN-Face-Recognition](https://github.com/PAC-P2P/BPNN-Face-Recognition)

Qt 版请戳：[Github . PAC-P2P/BPNN-Face-Recognition-For-Qt](https://github.com/PAC-P2P/BPNN-Face-Recognition-For-Qt)

## 安装依赖

### 安装 MPI

	# CentOS
	yum -y install openmpi openmpi-devel
	
	# 添加路径
	export PATH=$PATH:/usr/lib64/openmpi/bin/

### 安装 libcstl 库


	git clone https://github.com/activesys/libcstl.git
	
	cd libcstl
	
	# 配置环境，[指定安装路径]
	./configure [--prefix=newpath] 
	
	# 编译
	make
	
	# 安装
	make install
	
	# 检查
	make check
	
## 编译运行

	git clone https://github.com/PAC-P2P/BPNN-Face-Recognition-For-Parallel.git
	
	cd BPNN-Face-Recognition-For-Parallel
	
	# 解压训练集
	unzip data.zip
	
	cd src
	
	## 编译
	make
	
	# 运行，指定进程数为4
	mpirun -np 4 ./BPNN
	
	# 输入训练次数为100次
	100
	


## 数据集

[Neural Networks for Face Recognition](http://www.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/faces.html)
	