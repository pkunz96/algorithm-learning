import numpy

from algorithms.environment_oop import Environment

from typing import List





def top_down_merge_sort(env: Environment) -> List[int]:
    # Indexes, referencing elements in a list.
    stack_index_in_arr_low: int = env.bind_reg(0, 1)
    value_at_stack_index_in_arr_low: int = env.bind_arr(0, 1, 2)
    stack_index_in_arr_mid: int = env.bind_reg(1, 1)
    value_at_stack_index_in_arr_mid: int = env.bind_arr(1, 1, 2)
    stack_index_in_arr_high: int = env.bind_reg(2, 1)
    value_at_stack_index_in_arr_high: int = env.bind_arr(2, 1, 2)
    stack_var_first_half_sorted: int = env.bind_reg(3, 1)
    stack_var_second_half_sorted: int = env.bind_reg(4, 1)
    var_in_arr_old_mid_index: int = env.bind_reg(0, 0)
    var_in_arr_old_low_index: int = env.bind_reg(1, 0)
    var_in_arr_old_high_index: int = env.bind_reg(2, 0)
    var_old_first_half_sorted: int = env.bind_reg(3, 0)
    var_old_second_half_sorted: int = env.bind_reg(4, 0)
    index_in_arr_first_head: int = env.bind_reg(5, 0)
    value_at_index_in_arr_first_head: int = env.bind_arr(5, 0, 2)
    index_in_arr_second_head: int = env.bind_reg(6, 0)
    value_at_index_in_arr_second_head: int = env.bind_arr(6, 0, 2)
    var_out_arr_max_index: int = env.bind_reg(7, 0)
    index_in_arr_cur: int = env.bind_reg(8, 0)
    value_at_index_in_arr_cur: int = env.bind_arr(8, 0, 2)
    index_out_arr_cur: int = env.bind_reg(9, 0)
    value_at_index_out_arr_cur: int = env.bind_arr(9, 0, 3)
    #  Variables
    var_tmp: int = env.bind_reg(10, 0)
    var_second_head_minus_one: int = env.bind_reg(11, 0)
    env.set_program_counter(1)
    env.stack_register_file.push_register_file_stack()
    env.stack.push_address_type(1)
    env.stack.push_constant(0)
    env.stack.push_constant(0)
    env.stack.store()
    env.stack.push_address_type(1)
    env.stack.push_constant(1)
    env.stack.push_in_arr_length()
    env.stack.push_constant(2)
    env.stack.div()
    env.stack.push_constant(1)
    env.stack.sub()
    env.stack.store()
    env.stack.push_address_type(1)
    env.stack.push_constant(2)
    env.stack.push_in_arr_length()
    env.stack.push_constant(1)
    env.stack.sub()
    env.stack.store()
    while 2 <= len(env.stack_register_file.register_file_stack): # 20
        env.stack.push_address_type(0)
        env.stack.push_constant(1)
        env.stack.push_address_type(1)
        env.stack.push_constant(0)
        env.stack.push()
        env.stack.store()
        env.stack.push_address_type(0)
        env.stack.push_constant(0)
        env.stack.push_address_type(1)
        env.stack.push_constant(1)
        env.stack.push()
        env.stack.store()
        env.stack.push_address_type(0)
        env.stack.push_constant(2)
        env.stack.push_address_type(1)
        env.stack.push_constant(2)
        env.stack.push()
        env.stack.store()
        env.stack.push_address_type(0)
        env.stack.push_constant(3)
        env.stack.push_address_type(1)
        env.stack.push_constant(3)
        env.stack.push()
        env.stack.store()
        env.stack.push_address_type(0)
        env.stack.push_constant(4)
        env.stack.push_address_type(1)
        env.stack.push_constant(4)
        env.stack.push()
        env.stack.store()
        env.stack_register_file.pop_register_file_stack()
        if env.value_reg(var_old_first_half_sorted) <= 0: # 51
            env.set_program_counter(52)
            env.stack_register_file.push_register_file_stack()
            env.stack.push_address_type(1)
            env.stack.push_constant(1)
            env.stack.push_address_type(0)
            env.stack.push_constant(0)
            env.stack.push()
            env.stack.store()
            env.stack.push_address_type(1)
            env.stack.push_constant(0)
            env.stack.push_address_type(0)
            env.stack.push_constant(1)
            env.stack.push()
            env.stack.store()
            env.stack.push_address_type(1)
            env.stack.push_constant(2)
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.store()
            env.stack.push_address_type(1)
            env.stack.push_constant(3)
            env.stack.push_constant(1)
            env.stack.store()
            env.stack.push_address_type(1)
            env.stack.push_constant(4)
            env.stack.push_address_type(0)
            env.stack.push_constant(4)
            env.stack.push()
            env.stack.store()
            env.stack.push_address_type(0)
            env.stack.push_constant(10)
            env.stack.push_address_type(0)
            env.stack.push_constant(0)
            env.stack.push()
            env.stack.push_constant(1)
            env.stack.sub()
            env.stack.store()
            if env.value_reg(var_in_arr_old_low_index) <= env.value_reg(var_tmp): #89
                env.set_program_counter(90) # Savepoint
                env.stack_register_file.push_register_file_stack()
                env.stack.push_address_type(1)
                env.stack.push_constant(0)
                env.stack.push_address_type(0)
                env.stack.push_constant(1)
                env.stack.push()
                env.stack.store()
                env.stack.push_address_type(1)
                env.stack.push_constant(1)
                env.stack.push_address_type(0)
                env.stack.push_constant(1)
                env.stack.push()
                env.stack.push_address_type(0)
                env.stack.push_constant(0)
                env.stack.push()
                env.stack.push_address_type(0)
                env.stack.push_constant(1)
                env.stack.push()
                env.stack.sub()
                env.stack.push_constant(2)
                env.stack.div()
                env.stack.add()
                env.stack.store()
                env.stack.push_address_type(1)
                env.stack.push_constant(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(0)
                env.stack.push()
                env.stack.store()
                env.stack.push_address_type(1)
                env.stack.push_constant(3)
                env.stack.push_constant(0)
                env.stack.store()
                env.stack.push_address_type(1)
                env.stack.push_constant(4)
                env.stack.push_constant(0)
                env.stack.store()
            else:
                env.set_program_counter(127)
            env.set_program_counter(20) # 127
        else:
            env.set_program_counter(128)
            if env.value_reg(var_old_second_half_sorted) <= 0: #128
                env.set_program_counter(129)
                env.stack_register_file.push_register_file_stack()
                env.stack.push_address_type(1)
                env.stack.push_constant(1)
                env.stack.push_address_type(0)
                env.stack.push_constant(0)
                env.stack.push()
                env.stack.store()
                env.stack.push_address_type(1)
                env.stack.push_constant(0)
                env.stack.push_address_type(0)
                env.stack.push_constant(1)
                env.stack.push()
                env.stack.store()
                env.stack.push_address_type(1)
                env.stack.push_constant(2)
                env.stack.push_address_type(0)
                env.stack.push_constant(2)
                env.stack.push()
                env.stack.store()
                env.stack.push_address_type(1)
                env.stack.push_constant(3)
                env.stack.push_address_type(0)
                env.stack.push_constant(3)
                env.stack.push()
                env.stack.store()
                env.stack.push_address_type(1)
                env.stack.push_constant(4)
                env.stack.push_constant(1)
                env.stack.store()
                env.stack.push_address_type(0)
                env.stack.push_constant(10)
                env.stack.push_address_type(0)
                env.stack.push_constant(2)
                env.stack.push()
                env.stack.push_address_type(0)
                env.stack.push_constant(0)
                env.stack.push()
                env.stack.sub()
                env.stack.store()
                if 2 <= env.value_reg(var_tmp): # 168
                    env.set_program_counter(169)
                    env.stack_register_file.push_register_file_stack()
                    env.stack.push_address_type(1)
                    env.stack.push_constant(0)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(0)
                    env.stack.push()
                    env.stack.push_constant(1)
                    env.stack.add()
                    env.stack.store()
                    env.stack.push_address_type(1)
                    env.stack.push_constant(1)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(0)
                    env.stack.push()
                    env.stack.push_constant(1)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(2)
                    env.stack.push()
                    env.stack.push_address_type(0)
                    env.stack.push_constant(0)
                    env.stack.push()
                    env.stack.push_constant(1)
                    env.stack.sub()
                    env.stack.sub()
                    env.stack.push_constant(2)
                    env.stack.div()
                    env.stack.add()
                    env.stack.add()
                    env.stack.store()
                    env.stack.push_address_type(1)
                    env.stack.push_constant(2)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(2)
                    env.stack.push()
                    env.stack.store()
                    env.stack.push_address_type(1)
                    env.stack.push_constant(3)
                    env.stack.push_constant(0)
                    env.stack.store()
                    env.stack.push_address_type(1)
                    env.stack.push_constant(4)
                    env.stack.push_constant(0)
                    env.stack.store()
                else:
                    env.set_program_counter(212)
                env.set_program_counter(20)
            else:
                env.set_program_counter(213)
                env.stack.push_address_type(0)
                env.stack.push_constant(9)
                env.stack.push_constant(0)
                env.stack.store()
                env.stack.push_address_type(0)
                env.stack.push_constant(5)
                env.stack.push_address_type(0)
                env.stack.push_constant(1)
                env.stack.push()
                env.stack.store()
                env.stack.push_address_type(0)
                env.stack.push_constant(6)
                env.stack.push_address_type(0)
                env.stack.push_constant(0)
                env.stack.push()
                env.stack.push_constant(1)
                env.stack.add()
                env.stack.store()
                while env.value_reg(index_in_arr_first_head) <= env.value_reg(var_in_arr_old_mid_index) and env.value_reg(index_in_arr_second_head) <= env.value_reg(var_in_arr_old_high_index): # 231
                    env.set_program_counter(232)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(11)
                    env.stack.push_address_type(2)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(6)
                    env.stack.push()
                    env.stack.push()
                    env.stack.store()
                    if env.value_arr(value_at_index_in_arr_first_head) <= env.value_reg(var_second_head_minus_one): # 240
                        env.set_program_counter(241)
                        env.stack.push_address_type(3)
                        env.stack.push_address_type(0)
                        env.stack.push_constant(9)
                        env.stack.push()
                        env.stack.push_address_type(2)
                        env.stack.push_address_type(0)
                        env.stack.push_constant(5)
                        env.stack.push()
                        env.stack.push()
                        env.stack.store()
                        env.stack.push_address_type(0)
                        env.stack.push_constant(5)
                        env.stack.push_address_type(0)
                        env.stack.push_constant(5)
                        env.stack.push()
                        env.stack.push_constant(1)
                        env.stack.add()
                        env.stack.store()
                        env.stack.push_address_type(0)
                        env.stack.push_constant(9)
                        env.stack.push_address_type(0)
                        env.stack.push_constant(9)
                        env.stack.push()
                        env.stack.push_constant(1)
                        env.stack.add()
                        env.stack.store()
                        env.set_program_counter(293)
                    else:
                        env.set_program_counter(266)
                        env.stack.push_address_type(3)
                        env.stack.push_address_type(0)
                        env.stack.push_constant(9)
                        env.stack.push()
                        env.stack.push_address_type(2)
                        env.stack.push_address_type(0)
                        env.stack.push_constant(6)
                        env.stack.push()
                        env.stack.push()
                        env.stack.store()
                        env.stack.push_address_type(0)
                        env.stack.push_constant(6)
                        env.stack.push_address_type(0)
                        env.stack.push_constant(6)
                        env.stack.push()
                        env.stack.push_constant(1)
                        env.stack.add()
                        env.stack.store()
                        env.stack.push_address_type(0)
                        env.stack.push_constant(9)
                        env.stack.push_address_type(0)
                        env.stack.push_constant(9)
                        env.stack.push()
                        env.stack.push_constant(1)
                        env.stack.add()
                        env.stack.store()
                    env.set_program_counter(231)
                else:
                    env.set_program_counter(294)
                while env.value_reg(index_in_arr_first_head) <= env.value_reg(var_in_arr_old_mid_index): # 294
                    env.set_program_counter(295)
                    env.stack.push_address_type(3)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(9)
                    env.stack.push()
                    env.stack.push_address_type(2)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(5)
                    env.stack.push()
                    env.stack.push()
                    env.stack.store()
                    env.stack.push_address_type(0)
                    env.stack.push_constant(5)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(5)
                    env.stack.push()
                    env.stack.push_constant(1)
                    env.stack.add()
                    env.stack.store()
                    env.stack.push_address_type(0)
                    env.stack.push_constant(9)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(9)
                    env.stack.push()
                    env.stack.push_constant(1)
                    env.stack.add()
                    env.stack.store()
                    env.set_program_counter(294)
                else:
                    env.set_program_counter(322)
                while env.value_reg(index_in_arr_second_head) <= env.value_reg(var_in_arr_old_high_index): # 322
                    env.set_program_counter(323)
                    env.stack.push_address_type(3)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(9)
                    env.stack.push()
                    env.stack.push_address_type(2)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(6)
                    env.stack.push()
                    env.stack.push()
                    env.stack.store()
                    env.stack.push_address_type(0)
                    env.stack.push_constant(6)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(6)
                    env.stack.push()
                    env.stack.push_constant(1)
                    env.stack.add()
                    env.stack.store()
                    env.stack.push_address_type(0)
                    env.stack.push_constant(9)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(9)
                    env.stack.push()
                    env.stack.push_constant(1)
                    env.stack.add()
                    env.stack.store()
                    env.set_program_counter(322)
                else:
                    env.set_program_counter(249)
                env.stack.push_address_type(0)
                env.stack.push_constant(7)
                env.stack.push_address_type(0)
                env.stack.push_constant(9)
                env.stack.push()
                env.stack.push_constant(1)
                env.stack.sub()
                env.stack.store()
                env.stack.push_address_type(0)
                env.stack.push_constant(9)
                env.stack.push_constant(0)
                env.stack.store()
                env.stack.push_address_type(0)
                env.stack.push_constant(8)
                env.stack.push_address_type(0)
                env.stack.push_constant(1)
                env.stack.push()
                env.stack.store()
                while env.value_reg(index_out_arr_cur) <= env.value_reg(var_out_arr_max_index): # 267
                    env.set_program_counter(268)
                    env.stack.push_address_type(2)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(8)
                    env.stack.push()
                    env.stack.push_address_type(3)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(9)
                    env.stack.push()
                    env.stack.push()
                    env.stack.store()
                    env.stack.push_address_type(0)
                    env.stack.push_constant(8)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(8)
                    env.stack.push()
                    env.stack.push_constant(1)
                    env.stack.add()
                    env.stack.store()
                    env.stack.push_address_type(0)
                    env.stack.push_constant(9)
                    env.stack.push_address_type(0)
                    env.stack.push_constant(9)
                    env.stack.push()
                    env.stack.push_constant(1)
                    env.stack.add()
                    env.stack.store()
                    env.set_program_counter(267)
                else:
                    env.set_program_counter(295)
                env.set_program_counter(20)
    return env.in_arr;


def gen_top_down_merge_sort_environment(problem: List[int]) -> Environment:
    return Environment(problem, [0]*len(problem), [], 12, 5, include_stack_length=True)
