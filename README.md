import numpy as np
def compute_vandermode_matrix(xi):
  n=np.size([xi])
  vandermode_matrix=np.zeros([n,n])
  for i in range(n):
    for j in range(n):
      vandermode_matrix[i,j]=xi[i]**(n-j-1)
  return vandermode_matrix
#xi=[1,2,4]
#print(compute_vandermode_matrix(xi))


def vandermode_interpolation(vandermode_matrix,fi):
  coefficient_vector=np.zeros(np.size(xi))
  coefficient_vector=np.linalg.solve(vandermode_matrix,fi)
  return coefficient_vector
a=[[1,1,1],[4,2,1],[16,4,1]]
b=[2,5,2]
#print(vandermode_interpolation(a,b))


def evaluate_solution(evaluation_point,coefficient_vector):
  solution = 0.0
  n=np.size(coefficient_vector)
  for i in range(n):
    solution += coefficient_vector[i] * evaluation_point **(n-i-1)
  return solution
e=2
m=[1,2,3]
#print(evaluate_solution(e,m))


def f_1(x):
  return x**2


def f_2(x):
  return np.sin(x)


def f_3(x):
  fun = 1/(1 + 2 * (x**2))
  return fun
#print(f_3(0))


a=0
b=2


number_points_1=5
number_points_2=25
number_points_3=50
number_points_4=100


xi=np.linspace(a,b,number_points_1)
pi=np.linspace(a,b,number_points_2)
qi=np.linspace(a,b,number_points_3)
ri=np.linspace(a,b,number_points_4)


fi_1=f_1(xi)
fi_2=f_2(xi)
fi_3=f_3(xi)


#print(fi_1)
vandermode_matrix=np.zeros((np.size(xi),np.size(xi)))
vandermode_matrix=compute_vandermode_matrix(xi)
#print(vandermode_matrix)


coefficient_vector_1=np.zeros(np.size(xi))
coefficient_vector_2=np.zeros(np.size(xi))
coefficient_vector_3=np.zeros(np.size(xi))


coefficient_vector_1=vandermode_interpolation(vandermode_matrix,fi_1)
coefficient_vector_2=vandermode_interpolation(vandermode_matrix,fi_2)
coefficient_vector_3=vandermode_interpolation(vandermode_matrix,fi_3)
#print(coefficient_vector_1)


evaluation_point=np.pi/2


analytical_solution_1=f_1(evaluation_point)
analytical_solution_2=f_2(evaluation_point)
analytical_solution_3=f_3(evaluation_point)
#print(analytical_solution_1)


numerical_solution_1= evaluate_solution(evaluation_point,coefficient_vector_1)
numerical_solution_2= evaluate_solution(evaluation_point,coefficient_vector_2)
numerical_solution_3= evaluate_solution(evaluation_point,coefficient_vector_3)
#print(numerical_solution_1)


error_1= np.abs(analytical_solution_1 - numerical_solution_1)
error_2= np.abs(analytical_solution_2 - numerical_solution_2)
error_3= np.abs(analytical_solution_3 - numerical_solution_3)
#print(error_1,error_2,error_3)


print(f"N_points : {number_points_1}, f_1(x) = x^2, Numerical Solution : {numerical_solution_1}, Analytical Solution : {analytical_solution_1}, Error : {error_1}")
print(f"N_points : {number_points_1}, f_2(x) = sin(x), Numerical Solution : {numerical_solution_2}, Analytical Solution : {analytical_solution_2}, Error : {error_2}")
print(f"N_points : {number_points_1}, f_3(x) = 1/(1+2x^2), Numerical Solution : {numerical_solution_3}, Analytical solution : {analytical_solution_3}, Error : {error_3}")

#now,doing for n=25



fi_1=f_1(pi)
fi_2=f_2(pi)
fi_3=f_3(pi)


vandermode_matrix=np.zeros((np.size(pi),np.size(pi)))
vandermode_matrix=compute_vandermode_matrix(pi)
#print(vandermode_matrix)


coefficient_vector_1=np.zeros(np.size(pi))
coefficient_vector_2=np.zeros(np.size(pi))
coefficient_vector_3=np.zeros(np.size(pi))


coefficient_vector_1=vandermode_interpolation(vandermode_matrix,fi_1)
coefficient_vector_2=vandermode_interpolation(vandermode_matrix,fi_2)
coefficient_vector_3=vandermode_interpolation(vandermode_matrix,fi_3)
#print(coefficient_vector_1)


evaluation_point=np.pi/2


analytical_solution_1=f_1(evaluation_point)
analytical_solution_2=f_2(evaluation_point)
analytical_solution_3=f_3(evaluation_point)
#print(analytical_solution_1)


numerical_solution_1= evaluate_solution(evaluation_point,coefficient_vector_1)
numerical_solution_2= evaluate_solution(evaluation_point,coefficient_vector_2)
numerical_solution_3= evaluate_solution(evaluation_point,coefficient_vector_3)
#print(numerical_solution_1)


error_1= np.abs(analytical_solution_1 - numerical_solution_1)
error_2= np.abs(analytical_solution_2 - numerical_solution_2)
error_3= np.abs(analytical_solution_3 - numerical_solution_3)
#print(error_1,error_2,error_3)


print(f"N_points : {number_points_2}, f_1(x) = x^2, Numerical Solution : {numerical_solution_1}, Analytical Solution : {analytical_solution_1}, Error : {error_1}")
print(f"N_points : {number_points_2}, f_2(x) = sin(x), Numerical Solution : {numerical_solution_2}, Analytical Solution : {analytical_solution_2}, Error : {error_2}")
print(f"N_points : {number_points_2}, f_3(x) = 1/(1+2x^2), Numerical Solution : {numerical_solution_3}, Analytical solution : {analytical_solution_3}, Error : {error_3}")

#now, for n=50

fi_1=f_1(qi)
fi_2=f_2(qi)
fi_3=f_3(qi)


vandermode_matrix=np.zeros((np.size(qi),np.size(qi)))
vandermode_matrix=compute_vandermode_matrix(qi)
#print(vandermode_matrix)


coefficient_vector_1=np.zeros(np.size(qi))
coefficient_vector_2=np.zeros(np.size(qi))
coefficient_vector_3=np.zeros(np.size(qi))


coefficient_vector_1=vandermode_interpolation(vandermode_matrix,fi_1)
coefficient_vector_2=vandermode_interpolation(vandermode_matrix,fi_2)
coefficient_vector_3=vandermode_interpolation(vandermode_matrix,fi_3)
#print(coefficient_vector_1)


evaluation_point=np.pi/2


analytical_solution_1=f_1(evaluation_point)
analytical_solution_2=f_2(evaluation_point)
analytical_solution_3=f_3(evaluation_point)
#print(analytical_solution_1)


numerical_solution_1= evaluate_solution(evaluation_point,coefficient_vector_1)
numerical_solution_2= evaluate_solution(evaluation_point,coefficient_vector_2)
numerical_solution_3= evaluate_solution(evaluation_point,coefficient_vector_3)
#print(numerical_solution_1)


error_1= np.abs(analytical_solution_1 - numerical_solution_1)
error_2= np.abs(analytical_solution_2 - numerical_solution_2)
error_3= np.abs(analytical_solution_3 - numerical_solution_3)
#print(error_1,error_2,error_3)


print(f"N_points : {number_points_3}, f_1(x) = x^2, Numerical Solution : {numerical_solution_1}, Analytical Solution : {analytical_solution_1}, Error : {error_1}")
print(f"N_points : {number_points_3}, f_2(x) = sin(x), Numerical Solution : {numerical_solution_2}, Analytical Solution : {analytical_solution_2}, Error : {error_2}")
print(f"N_points : {number_points_3}, f_3(x) = 1/(1+2x^2), Numerical Solution : {numerical_solution_3}, Analytical solution : {analytical_solution_3}, Error : {error_3}")

#now,for n=100

fi_1=f_1(ri)
fi_2=f_2(ri)
fi_3=f_3(ri)


vandermode_matrix=np.zeros((np.size(ri),np.size(ri)))
vandermode_matrix=compute_vandermode_matrix(ri)
#print(vandermode_matrix)


coefficient_vector_1=np.zeros(np.size(ri))
coefficient_vector_2=np.zeros(np.size(ri))
coefficient_vector_3=np.zeros(np.size(ri))


coefficient_vector_1=vandermode_interpolation(vandermode_matrix,fi_1)
coefficient_vector_2=vandermode_interpolation(vandermode_matrix,fi_2)
coefficient_vector_3=vandermode_interpolation(vandermode_matrix,fi_3)
#print(coefficient_vector_1)


evaluation_point=np.pi/2


analytical_solution_1=f_1(evaluation_point)
analytical_solution_2=f_2(evaluation_point)
analytical_solution_3=f_3(evaluation_point)
#print(analytical_solution_1)


numerical_solution_1= evaluate_solution(evaluation_point,coefficient_vector_1)
numerical_solution_2= evaluate_solution(evaluation_point,coefficient_vector_2)
numerical_solution_3= evaluate_solution(evaluation_point,coefficient_vector_3)
#print(numerical_solution_1)


error_1= np.abs(analytical_solution_1 - numerical_solution_1)
error_2= np.abs(analytical_solution_2 - numerical_solution_2)
error_3= np.abs(analytical_solution_3 - numerical_solution_3)
#print(error_1,error_2,error_3)


print(f"N_points : {number_points_4}, f_1(x) = x^2, Numerical Solution : {numerical_solution_1}, Analytical Solution : {analytical_solution_1}, Error : {error_1}")
print(f"N_points : {number_points_4}, f_2(x) = sin(x), Numerical Solution : {numerical_solution_2}, Analytical Solution : {analytical_solution_2}, Error : {error_2}")
print(f"N_points : {number_points_4}, f_3(x) = 1/(1+2x^2), Numerical Solution : {numerical_solution_3}, Analytical solution : {analytical_solution_3}, Error : {error_3}")


