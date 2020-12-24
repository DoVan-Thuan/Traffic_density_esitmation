from fuzzy_system.fuzzy_variable_output import FuzzyOutputVariable
from fuzzy_system.fuzzy_variable_input import FuzzyInputVariable
# from fuzzy_system.fuzzy_variable import FuzzyVariable
from fuzzy_system.fuzzy_system import FuzzySystem

density = FuzzyInputVariable('Density', 0, 15000, 100)
density.add_triangular('Sparse', 0, 0, 10000)
density.add_triangular('Normal')
density.add_triangular('Crowd', 5000, 15000, 15000)
# density.add_triangular('Hot', 25, 40, 40)

# humidity = FuzzyInputVariable('Humidity', 20, 100, 100)
# humidity.add_triangular('Wet', 20, 20, 60)
# humidity.add_trapezoidal('Normal', 30, 50, 70, 90)
# humidity.add_triangular('Dry', 60, 100, 100)

light_time = FuzzyOutputVariable('Light_time', 0, 100, 100)
light_time.add_triangular('Short', 0, 0, 60)
light_time.add_triangular('Normal')
light_time.add_triangular('Long', 30, 100, 100)
# motor_speed.add_triangular('Fast', 50, 100, 100)

system = FuzzySystem()
system.add_input_variable(density)
# system.add_input_variable(humidity)
system.add_output_variable(light_time)

system.add_rule(
		{ 'Density':'Sparse'},
		{ 'Light_time':'Long'})

system.add_rule(
		{'Density': 'Crowd'},
		{'Light_time': 'Short'})


print(output)
# print('fuzzification\n-------------\n', info['fuzzification'])
# print('rules\n-----\n', info['rules'])

system.plot_system()