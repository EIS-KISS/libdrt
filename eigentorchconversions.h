#include <climits>
#include <sys/types.h>
#include <torch/torch.h>
#include <Eigen/Dense>
#include <torch/types.h>
#include <vector>

template <typename V>
inline torch::TensorOptions tensorOptCpuNg()
{
	static_assert(std::is_same<V, float>::value || std::is_same<V, double>::value,
				  "This function can only be passed double or float types");
	torch::TensorOptions options;
	if constexpr(std::is_same<V, float>::value)
		options = options.dtype(torch::kFloat32);
	else
		options = options.dtype(torch::kFloat64);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	options = options.requires_grad(false);
	return options;
}

template <typename V>
bool checkTorchType(const torch::Tensor& tensor)
{
	static_assert(std::is_same<V, float>::value || std::is_same<V, double>::value ||
		std::is_same<V, int64_t>::value || std::is_same<V, int32_t>::value || std::is_same<V, int8_t>::value,
				  "This function dose not work with this type");
	if constexpr(std::is_same<V, float>::value)
		return tensor.dtype() == torch::kFloat32;
	else if constexpr(std::is_same<V, double>::value)
		return tensor.dtype() == torch::kFloat64;
	else if constexpr(std::is_same<V, int64_t>::value)
		return tensor.dtype() == torch::kInt64;
	else if constexpr(std::is_same<V, int32_t>::value)
		return tensor.dtype() == torch::kInt32;
	else if constexpr(std::is_same<V, int8_t>::value)
		return tensor.dtype() == torch::kInt8;
}

template <typename V>
using MatrixXrm = typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename V>
torch::Tensor eigen2libtorch(Eigen::MatrixX<V> &M)
{
	Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> E(M);
	std::vector<int64_t> dims = {E.rows(), E.cols()};
	auto T = torch::from_blob(E.data(), dims, tensorOptCpuNg<V>()).clone();
	return T;
}

template <typename V>
torch::Tensor eigen2libtorch(MatrixXrm<V> &E, bool copydata = true)
{
	std::vector<int64_t> dims = {E.rows(), E.cols()};
	auto T = torch::from_blob(E.data(), dims, tensorOptCpuNg<V>());
	if (copydata)
		return T.clone();
	else
		return T;
}

template<typename V>
Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic> libtorch2eigenMaxtrix(torch::Tensor &Tin)
{
	/*
	LibTorch is Row-major order and Eigen is Column-major order.
	MatrixXrm uses Eigen::RowMajor for compatibility.
	*/
	assert(checkTorchType<V>(Tin));
	auto T = Tin.to(torch::kCPU);
	Eigen::Map<MatrixXrm<V>> E(T.data_ptr<V>(), T.size(0), T.size(1));
	return E;
}

template<typename V>
Eigen::Vector<V, Eigen::Dynamic> libtorch2eigenVector(torch::Tensor &Tin)
{
	assert(Tin.sizes().size() == 1);
	assert(checkTorchType<V>(Tin));
	auto T = Tin.to(torch::kCPU);
	Eigen::Map<Eigen::Vector<V, Eigen::Dynamic>> E(T.data_ptr<V>(), T.numel());
	return E;
}
