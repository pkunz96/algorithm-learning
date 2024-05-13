import numpy

from algorithms.environment_oop import Environment

from typing import List



def heap_sort(env: Environment) -> List[int]:

    # Indexes, referencing elements in a list.

    index_in_arr_high: int = env.bind_reg(0, 0)

    value_at_index_in_arr_high: int = env.bind_arr(0, 0, 2)

    index_in_arr_sink: int = env.bind_reg(1, 0)

    value_at_index_in_arr_sink: int = env.bind_arr(1, 0, 2)

    index_in_arr_parent: int = env.bind_reg(2, 0)

    value_at_index_in_arr_parent: int = env.bind_arr(2, 0, 2)

    index_in_arr_left_child: int = env.bind_reg(3, 0)

    value_at_index_in_arr_left_child: int = env.bind_arr(3, 0, 2)

    index_in_arr_right_child: int = env.bind_reg(4, 0)

    value_at_index_in_arr_right_child: int = env.bind_arr(4, 0, 2)

    index_in_arr_max: int = env.bind_reg(5, 0)

    value_at_index_in_arr_max: int = env.bind_arr(5, 0, 2)

    index_in_arr_first: int = env.bind_reg(6, 0)

    value_at_index_in_arr_first: int = env.bind_arr(6, 0, 2)

    # Variables

    var_tmp: int = env.bind_reg(7, 0)

    var_sd_parent_of_largest: int = env.bind_reg(8, 0)

    var_sd_parent_2_i_of_largest: int = env.bind_reg(9, 0)

    stack_var: int = env.bind_reg(0, 1)

    var_left_child_minus_one: int = env.bind_reg(10, 0)
    var_right_child_minus_one: int = env.bind_reg(11, 0)

    # Start
    env.stack.push_address_type(0)
    env.stack.push_constant(0)
    env.stack.push_in_arr_length()
    env.stack.push_constant(1)
    env.stack.sub()
    env.stack.store()
    env.stack.push_address_type(0)
    env.stack.push_constant(1)
    env.stack.push_in_arr_length()
    env.stack.push_constant(2)
    env.stack.div()
    env.stack.store()
    env.stack.push_address_type(0)
    env.stack.push_constant(6)
    env.stack.push_constant(0)
    env.stack.store()
    while 0 <= env.value_reg(index_in_arr_sink):
        env.set_program_counter(17)
        env.stack_register_file.push_register_file_stack()
        env.stack.push_address_type(1)
        env.stack.push_constant(0)
        env.stack.push_address_type(0)
        env.stack.push_constant(1)
        env.stack.push()
        env.stack.store()
        while 2 <= len(env.stack_register_file.register_file_stack):
            env.set_program_counter(25)
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push_address_type(1)
            env.stack.push_constant(0)
            env.stack.push()
            env.stack.store()
            env.stack_register_file.pop_register_file_stack()
            env.stack.push_address_type(0)
            env.stack.push_constant(3)
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.push_constant(1)
            env.stack.add()
            env.stack.add()
            env.stack.store()
            env.stack.push_address_type(0)
            env.stack.push_constant(4)
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.push_constant(2)
            env.stack.add()
            env.stack.add()
            env.stack.store()
            env.stack.push_address_type(0)
            env.stack.push_constant(5)
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.store()
            if env.value_reg(index_in_arr_left_child) <= env.value_reg(index_in_arr_high):
                env.set_program_counter(63)
                env.stack.push_address_type(0)
                env.stack.push_constant(10)
                env.stack.push_address_type(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(3)
                env.stack.push()
                env.stack.push()
                env.stack.push_constant(1)
                env.stack.sub()
                env.stack.store()
                if env.value_arr(value_at_index_in_arr_parent) <= env.value_reg(var_left_child_minus_one):
                    env.stack.push_address_type(0)
                    env.stack.push_constant(5)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(3)
                    env.stack.push()
                    env.stack.store()
                else:
                    env.set_program_counter(79)
            if env.value_reg(index_in_arr_right_child) <= env.value_reg(index_in_arr_high):
                env.set_program_counter(80)
                env.stack.push_address_type(0)
                env.stack.push_constant(11)
                env.stack.push_address_type(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(4)
                env.stack.push()
                env.stack.push()
                env.stack.push_constant(1)
                env.stack.sub()
                env.stack.store()
                if env.value_arr(value_at_index_in_arr_max) <= env.value_reg(var_right_child_minus_one): # 91
                    env.set_program_counter(91)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(5)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(4)
                    env.stack.push()
                    env.stack.store()
                else:
                    env.set_program_counter(97)
            else:
                env.set_program_counter(97)
            env.stack.push_address_type(0)
            env.stack.push_constant(8)
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.push_address_type(0)
            env.stack.push_constant(5)
            env.stack.push()
            env.stack.sub()
            env.stack.square()
            env.stack.push_constant(1)
            env.stack.sub()
            env.stack.store()
            if 0 <= env.value_reg(var_sd_parent_of_largest): # 111
                env.set_program_counter(111)
                env.stack.push_address_type(0)
                env.stack.push_constant(7)
                env.stack.push_address_type(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(2)
                env.stack.push()
                env.stack.push()
                env.stack.store()
                env.stack.push_address_type(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(2)
                env.stack.push()
                env.stack.push_address_type(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(5)
                env.stack.push()
                env.stack.push()
                env.stack.store()
                env.stack.push_address_type(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(5)
                env.stack.push()
                env.stack.push_address_type(0)
                env.stack.push_constant(7)
                env.stack.push()
                env.stack.store()
                env.stack_register_file.push_register_file_stack()
                env.stack.push_address_type(1)
                env.stack.push_constant(0)
                env.stack.push_address_type(0)
                env.stack.push_constant(5)
                env.stack.push()
                env.stack.store()
            else:
                env.set_program_counter(144)
            env.set_program_counter(24)
        else:
            env.set_program_counter(145)
        env.stack.push_address_type(0)
        env.stack.push_constant(1)
        env.stack.push_address_type(0)
        env.stack.push_constant(1)
        env.stack.push()
        env.stack.push_constant(1)
        env.stack.sub()
        env.stack.store()
        env.set_program_counter(16)
    else:
        env.set_program_counter(153)
    env.stack.push_address_type(0)
    env.stack.push_constant(1)
    env.stack.push_address_type(0)
    env.stack.push_constant(0)
    env.stack.push()
    env.stack.store()
    while 0 <= env.value_reg(index_in_arr_sink):
        env.set_program_counter(160)
        env.stack.push_address_type(0)
        env.stack.push_constant(7)
        env.stack.push_address_type(2)
        env.stack.push_address_type(0)
        env.stack.push_constant(6)
        env.stack.push()
        env.stack.push()
        env.stack.store()
        env.stack.push_address_type(2)
        env.stack.push_address_type(0)
        env.stack.push_constant(6)
        env.stack.push()
        env.stack.push_address_type(2)
        env.stack.push_address_type(0)
        env.stack.push_constant(1)
        env.stack.push()
        env.stack.push()
        env.stack.store()
        env.stack.push_address_type(2)
        env.stack.push_address_type(0)
        env.stack.push_constant(1)
        env.stack.push()
        env.stack.push_address_type(0)
        env.stack.push_constant(7)
        env.stack.push()
        env.stack.store()
        env.stack.push_address_type(0)
        env.stack.push_constant(0)
        env.stack.push_address_type(0)
        env.stack.push_constant(0)
        env.stack.push()
        env.stack.push_constant(1)
        env.stack.sub()
        env.stack.store()
        env.stack_register_file.push_register_file_stack()
        env.stack.push_address_type(1)
        env.stack.push_constant(0)
        env.stack.push_constant(0)
        env.stack.store()
        while 2 <= len(env.stack_register_file.register_file_stack):
            env.set_program_counter(200) # 200
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push_address_type(1)
            env.stack.push_constant(0)
            env.stack.push()
            env.stack.store()
            env.stack_register_file.pop_register_file_stack()
            env.stack.push_address_type(0)
            env.stack.push_constant(3)
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.push_constant(1)
            env.stack.add()
            env.stack.add()
            env.stack.store()
            env.stack.push_address_type(0)
            env.stack.push_constant(4)
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.push_constant(2)
            env.stack.add()
            env.stack.add()
            env.stack.store()
            env.stack.push_address_type(0)
            env.stack.push_constant(5)
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.store()
            if env.value_reg(index_in_arr_left_child) <= env.value_reg(index_in_arr_high):
                env.set_program_counter(238)
                env.stack.push_address_type(0)
                env.stack.push_constant(10)
                env.stack.push_address_type(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(3)
                env.stack.push()
                env.stack.push()
                env.stack.push_constant(1)
                env.stack.sub()
                env.stack.store()
                if env.value_arr(value_at_index_in_arr_parent) <= env.value_reg(var_left_child_minus_one):
                    env.set_program_counter(249)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(5)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(3)
                    env.stack.push()
                    env.stack.store()
                else:
                    env.set_program_counter(255)
            else:
                env.set_program_counter(255)
            if env.value_reg(index_in_arr_right_child) <= env.value_reg(index_in_arr_high):
                env.set_program_counter(256)
                env.stack.push_address_type(0)
                env.stack.push_constant(11)
                env.stack.push_address_type(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(4)
                env.stack.push()
                env.stack.push()
                env.stack.push_constant(1)
                env.stack.sub()
                env.stack.store()
                if env.value_arr(value_at_index_in_arr_max) <= env.value_reg(var_right_child_minus_one):
                    env.set_program_counter(267)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(5)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(4)
                    env.stack.push()
                    env.stack.store()
                else:
                    env.set_program_counter(273)
            else:
                env.set_program_counter(273)
            env.stack.push_address_type(0)
            env.stack.push_constant(9)
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.push_address_type(0)
            env.stack.push_constant(5)
            env.stack.push()
            env.stack.sub()
            env.stack.square()
            env.stack.push_constant(1)
            env.stack.sub()
            env.stack.store()
            if 0 <= env.value_reg(var_sd_parent_2_i_of_largest):
                env.set_program_counter(287)
                env.stack.push_address_type(0)
                env.stack.push_constant(7)
                env.stack.push_address_type(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(2)
                env.stack.push()
                env.stack.push()
                env.stack.store()
                env.stack.push_address_type(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(2)
                env.stack.push()
                env.stack.push_address_type(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(5)
                env.stack.push()
                env.stack.push()
                env.stack.store()
                env.stack.push_address_type(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(5)
                env.stack.push()
                env.stack.push_address_type(0)
                env.stack.push_constant(7)
                env.stack.push()
                env.stack.store()
                env.stack_register_file.push_register_file_stack()
                env.stack.push_address_type(1)
                env.stack.push_constant(0)
                env.stack.push_address_type(0)
                env.stack.push_constant(5)
                env.stack.push()
                env.stack.store()
            else:
                env.set_program_counter(320)
            env.set_program_counter(199)
        else:
            env.set_program_counter(321)
        env.stack.push_address_type(0)
        env.stack.push_constant(1)
        env.stack.push_address_type(0)
        env.stack.push_constant(1)
        env.stack.push()
        env.stack.push_constant(1)
        env.stack.sub()
        env.stack.store()
        env.set_program_counter(159)
    return env.in_arr


def gen_heap_sort_environment(problem: List[int]) -> Environment:
    return Environment(problem, [], [], 12, 1, include_stack_length = True)



heap_sort(gen_heap_sort_environment([3, 2, 1]))