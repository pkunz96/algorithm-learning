from typing import List
from algorithms.environment_oop import Environment


def straight_insertion_sort(env: Environment) -> List[int]:
    index_in_arr_next: int = env.bind_reg(0, 0)
    value_at_index_in_arr_cur_src: int = env.bind_arr(1, 0, 2)
    index_in_arr_insert: int = env.bind_reg(2, 0)
    value_at_index_in_arr_insert: int = env.bind_arr(2, 0, 2)
    var_max_in_arr_index: int = env.bind_reg(4, 0)
    value_at_index_in_arr_next: int = env.bind_arr(0, 0, 2)
    index_in_arr_cur_src: int = env.bind_reg(1, 0)
    var_tmp: int = env.bind_reg(3, 0)
    value_at_var_max_in_arr_index: int = env.bind_arr(4, 0, 2)
    # Start
    env.set_program_counter(1)
    env.stack.push_address_type(0)
    env.stack.push_constant(0)
    env.stack.push_constant(1)
    env.stack.store()
    env.stack.push_address_type(0)
    env.stack.push_constant(1)
    env.stack.push_address_type(0)
    env.stack.push_constant(0)
    env.stack.push()
    env.stack.store()
    env.stack.push_address_type(0)
    env.stack.push_constant(4)
    env.stack.push_in_arr_length()
    env.stack.push_constant(1)
    env.stack.sub()
    env.stack.store()
    while env.value_reg(index_in_arr_next) <= env.value_reg(var_max_in_arr_index):
        env.stack.push_address_type(0)
        env.stack.push_constant(2)
        env.stack.push_address_type(0)
        env.stack.push_constant(0)
        env.stack.push()
        env.stack.push_constant(1)
        env.stack.sub()
        env.stack.store()
        while 0 <= env.value_reg(index_in_arr_insert) and env.value_arr(value_at_index_in_arr_cur_src) <= env.value_arr(value_at_index_in_arr_insert):
            env.stack.push_address_type(0)
            env.stack.push_constant(3)
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
            env.stack.push_address_type(0)
            env.stack.push_constant(3)
            env.stack.push()
            env.stack.store()
            env.stack.push_address_type(0)
            env.stack.push_constant(1)
            env.stack.push_address_type(0)
            env.stack.push_constant(1)
            env.stack.push()
            env.stack.push_constant(1)
            env.stack.sub()
            env.stack.store()
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push_address_type(0)
            env.stack.push_constant(2)
            env.stack.push()
            env.stack.push_constant(1)
            env.stack.sub()
            env.stack.store()
            env.set_program_counter(25)
        else:
            env.set_program_counter(71)
        env.stack.push_address_type(0)
        env.stack.push_constant(0)
        env.stack.push_address_type(0)
        env.stack.push_constant(0)
        env.stack.push()
        env.stack.push_constant(1)
        env.stack.add()
        env.stack.store()
        env.stack.push_address_type(0)
        env.stack.push_constant(1)
        env.stack.push_address_type(0)
        env.stack.push_constant(0)
        env.stack.push()
        env.stack.store()
        env.set_program_counter(17)
    env.set_program_counter(86)
    return env.in_arr


def gen_insertion_sort_environment(problem: List[int]) -> Environment:
    return Environment(problem, [], [], 5, 0)
