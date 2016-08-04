#include <cstdio>
#include <thread>
#include "tbb/flow_graph.h"
#include "TH/TH.h"
#include <array>
#include <tuple>

// This is an op that gets added to the TBB graph
class Cadd {
 public:
  static const int kNumInputs_ = 2;
  static const int kNumOutputs_ = 1;
 private:
  const std::array<bool, kNumInputs_> requires_gradient_;
  const std::array<bool, kNumInputs_> allow_inplace_;
 public:
  typedef std::tuple<THDoubleTensor*, THDoubleTensor*> input_types_;
  typedef std::tuple<THDoubleTensor*> output_types_;
  Cadd(std::array<bool, kNumInputs_> requires_gradient,
       std::array<bool, kNumInputs_> allow_inplace)
    : requires_gradient_(requires_gradient), allow_inplace_(allow_inplace) {}
  THDoubleTensor* operator()(input_types_ v) {
    THDoubleTensor* result = THDoubleTensor_new();
    THDoubleTensor_cadd(result, tbb::flow::get<0>(v), 1, tbb::flow::get<1>(v));
    return result;
  }
};

// This is the node that Python gets a handle to
template<class data_type>
class ADNode {
 public:
  tbb::flow::write_once_node<data_type>* node_;
  data_type output_; // Scalar/tensor int/float/double
  const bool requires_gradient_;
  ~ADNode() { printf("Being destroyed!\n"); }
  ADNode(bool requires_gradient, tbb::flow::write_once_node<data_type>* node)
    : requires_gradient_(requires_gradient), node_(node) {};
  data_type get() {
    printf("Getting result!\n");
    while (!node_->try_get(output_))
      printf(".");
      std::this_thread::yield();
    printf("\n");
    return output_;
  }
};

// A function that takes a series of input nodes and an op, which it then
// schedules on the given graph
template<class op>
ADNode<THDoubleTensor*>* dispatch(
    std::tuple<ADNode<THDoubleTensor*>*, ADNode<THDoubleTensor*>*> inputs,
    std::array<bool, op::kNumInputs_> allow_inplace, tbb::flow::graph* g) {
  std::array<bool, op::kNumInputs_> requires_gradient;
  requires_gradient[0] = std::get<0>(inputs)->requires_gradient_;
  requires_gradient[1] = std::get<1>(inputs)->requires_gradient_;
  // Join
  tbb::flow::join_node<Cadd::input_types_, tbb::flow::queueing>* join = new tbb::flow::join_node<Cadd::input_types_, tbb::flow::queueing>(*g);
  // Op
  op* op_node = new op(requires_gradient, allow_inplace);
  tbb::flow::function_node<Cadd::input_types_, Cadd::output_types_>* node = new tbb::flow::function_node<Cadd::input_types_, Cadd::output_types_>(*g, tbb::flow::serial, *op_node);
  // Split
  tbb::flow::split_node<Cadd::output_types_>* split = new tbb::flow::split_node<Cadd::output_types_>(*g);
  // Save
  // TODO Find a way to not hardcode this!
  tbb::flow::write_once_node<std::tuple_element<0, Cadd::output_types_>::type>* write0 = new tbb::flow::write_once_node<std::tuple_element<0, Cadd::output_types_>::type>
      (*g);

  // Now add edges in reverse order (to make sure messages aren't discarded)
  make_edge(tbb::flow::output_port<0>(*split), *write0);
  make_edge(*node, *split);
  make_edge(*join, *node);
  make_edge(*std::get<0>(inputs)->node_, tbb::flow::input_port<0>(*join));
  make_edge(*std::get<1>(inputs)->node_, tbb::flow::input_port<1>(*join));

  ADNode<THDoubleTensor*>* ad_node = new ADNode<THDoubleTensor*>(std::any_of(requires_gradient.cbegin(), requires_gradient.cend(), [](bool x) {return x;}), write0);

  return ad_node;
};

ADNode<THDoubleTensor*>* create_node(tbb::flow::graph* g) {
  THDoubleTensor* tensor = THDoubleTensor_newWithSize1d(10);
  THDoubleStorage_fill(tensor->storage, 2);
  // Use new so that the nodes don't get destroyed at the end of the function
  tbb::flow::broadcast_node<THDoubleTensor*>* input = new tbb::flow::broadcast_node<THDoubleTensor*>(*g);
  tbb::flow::write_once_node<THDoubleTensor*>* node = new tbb::flow::write_once_node<THDoubleTensor*>(*g);
  make_edge(*input, *node);
  input->try_put(tensor);
  ADNode<THDoubleTensor*>* ad_node = new ADNode<THDoubleTensor*>(true, node);
  return ad_node;
}

int main() {
  tbb::flow::graph g;
  ADNode<THDoubleTensor*>* ad_node1 = create_node(&g);
  ADNode<THDoubleTensor*>* ad_node2 = create_node(&g);
  std::array<bool, 2> allow_inplace = {false, false};
  ADNode<THDoubleTensor*>* adnode3 = dispatch<Cadd>(std::make_tuple(ad_node1, ad_node2), allow_inplace, &g);
  printf("Got output node\n");
  printf("Graph done\n");
  THDoubleTensor* tensor = adnode3->get();
  printf("Got result\n");
  double* result(tensor->storage->data);
  printf("Final result is %f\n", *result);
  return 0;
}
