# Variables
NVCC = nvcc
CUDA_DIR = cuda/bfs
DATA_DIR = data/bfs
SRC_FILES = bfs.cu bfs_managed.cu bfs_time.cu bfs_managed_time.cu bfs_memory.cu bfs_managed_memory.cu
EXECUTABLES = $(SRC_FILES:.cu=)

# Compilation rules
all: $(addprefix $(CUDA_DIR)/, $(EXECUTABLES))

$(CUDA_DIR)/%: $(CUDA_DIR)/%.cu
	@echo "-------------------------------"
	@echo "Compiling $<"
	$(NVCC) -o $@ $<
	@echo "Giving permissions to $@"
	chmod +x $@

# Execution rules
run: all
	@echo "-------------------------------"
	@echo "Executing bfs.cu with graph1MW_6.txt"
	cd $(CUDA_DIR) && ./bfs ../../$(DATA_DIR)/graph1MW_6.txt

	@echo "-------------------------------"
	@echo "Executing bfs_managed.cu with graph1MW_6.txt"
	cd $(CUDA_DIR) && ./bfs_managed ../../$(DATA_DIR)/graph1MW_6.txt

	@echo "-------------------------------"
	@echo "Executing bfs_time.cu with all input files"
	cd $(CUDA_DIR) && ./bfs_time ../../$(DATA_DIR)/graph1MW_6.txt
	cd $(CUDA_DIR) && ./bfs_time ../../$(DATA_DIR)/graph4096.txt
	cd $(CUDA_DIR) && ./bfs_time ../../$(DATA_DIR)/graph65536.txt

	@echo "-------------------------------"
	@echo "Executing bfs_managed_time.cu with all input files"
	cd $(CUDA_DIR) && ./bfs_managed_time ../../$(DATA_DIR)/graph1MW_6.txt
	cd $(CUDA_DIR) && ./bfs_managed_time ../../$(DATA_DIR)/graph4096.txt
	cd $(CUDA_DIR) && ./bfs_managed_time ../../$(DATA_DIR)/graph65536.txt

	@echo "-------------------------------"
	@echo "Executing bfs_memory.cu with graph1MW_6.txt"
	cd $(CUDA_DIR) && ./bfs_memory ../../$(DATA_DIR)/graph1MW_6.txt

	@echo "-------------------------------"
	@echo "Executing bfs_managed_memory.cu with graph1MW_6.txt"
	cd $(CUDA_DIR) && ./bfs_managed_memory ../../$(DATA_DIR)/graph1MW_6.txt

# Cleaning generated files
clean:
	rm -f $(addprefix $(CUDA_DIR)/, $(EXECUTABLES)) newresult.txt
