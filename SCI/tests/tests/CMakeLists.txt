
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)


set(MATPLOTLIB_CPP_DIR "${CMAKE_SOURCE_DIR}/matplotlib-cpp")


include_directories(${MATPLOTLIB_CPP_DIR})
macro (add_test_OT _name)
    add_executable(${_name}-OT "test_ring_${_name}.cpp")
    target_include_directories(${_name}-OT PRIVATE ${MATPLOTLIB_CPP_DIR})
    target_link_libraries(${_name}-OT SCI-OT Python3::Python Python3::NumPy)
endmacro()


#add_test_OT(matmul)
#add_test_OT(value_extension)
#add_test_OT(truncation)
#add_test_OT(relu)
#add_test_OT(argmax)
#add_test_OT(aux_protocols)
#add_test_OT(maxpool)
#add_test_OT(hadamard_product)
#add_test_OT(sigmoid)
#add_test_OT(exp)
#add_test_OT(tanh)
#add_test_OT(sqrt)
#add_test_OT(multest)
#add_test_OT(floattest)
#add_test_OT(ABSMUX)
#add_test_OT(papertest)
#add_test_OT(l21_la6)
#add_test_OT(clear_l21_la6)
#add_test_OT(PosandNeg_l21_la6)
#add_test_OT(PosandNeg_l21_la6_test)
#add_test_OT(l21_la12)
#add_test_OT(auto_test)
#add_test_OT(single_test)
#add_test_OT(three_division)
#add_test_OT(gelu_new)
#add_test_OT(max_distance)
add_test_OT(threeinterval)
#add_test_OT(million)
add_test_OT(12_22_GELU_final)
add_test_OT(1_1_GELU_muilt_threads)
add_test_OT(12_23_ELU_final)
add_test_OT(1_8_ELU_muilt_threads)
add_test_OT(12_24_tanh_final)
add_test_OT(1_7_Tanh_muilt_threads)
add_test_OT(12_25_sigmoid_final)
add_test_OT(1_8_sigmoid_muilt_threads)

#add_test_HE(relu)
#add_test_HE(maxpool)
#add_test_HE(argmax)
#add_test_HE(conv)
#add_test_HE(fc)
#add_test_HE(elemwise_prod)

#add_test_float(bench_op)
#add_test_float(primitive)
#add_test_float(compound)



add_subdirectory(GC)


#add_test_floatml(vsum)
#add_test_floatml(dotprod)
#add_test_floatml(matmul)
#add_test_floatml(sigmoid)
#add_test_floatml(softmax)