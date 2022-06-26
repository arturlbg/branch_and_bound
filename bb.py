from mip import Model, MAXIMIZE, xsum
from numpy import asarray, abs
import re

class BB:
  def __init__(self):
    self.primal = 0
    self.optimal_model = None

  def read_txt(self, filePath: str):

    l = []
    with open(filePath) as file:
      for line in file:
          for i in re.findall(r'\d+', line):
              l.append(i)

    variables = int(l[0])
    rests = int(l[1])
    coef_obj = []
    for i in range(2, 2+variables):
      coef_obj.append(int(l[i]))

    count = 0
    aux = []
    coef_rests = []
    for i in range(2+variables, len(l)):
      if count == variables:
        aux.append(int(l[i]))
        coef_rests.append(aux)
        aux = []
        count = 0
      else:
        aux.append(int(l[i]))
        count += 1
    return variables, rests, coef_obj, coef_rests

  def solver(self, model: Model):
    model.optimize()
    params = {}
    params["objective"] = model.objective_value
    params["variables"] = model.vars

    return params

  def branch_and_bound(self, model: Model):
    nodes = [model]

    while nodes != []:
      model_solver = self.solver(nodes[0])
      aux = self.bound(nodes[0])
      if aux == 'INVIABLE' or aux == 'LIMIT':
        nodes.pop(0)
      elif aux == 'INTEGER':
        if model_solver["objective"] >= self.primal:
          self.optimal_model = nodes[0]
          self.primal = model_solver["objective"]
        nodes.pop(0)
      elif aux == 'VIABLE':
        branches = self.branch(nodes[0], model_solver["variables"])
        nodes.append(branches[0])
        nodes.append(branches[1])
        nodes.pop(0)

  def bound(self, model: Model):
    aux_solver = self.solver(model)
    count_int = 0

    if aux_solver["objective"] == None:
      return 'INVIABLE'

    for i in aux_solver["variables"]:
      if i.x.is_integer():
        count_int += 1

    if count_int == len(aux_solver["variables"]):
      return 'INTEGER'

    if aux_solver["objective"] <= self.primal:
      return 'LIMIT'
    return 'VIABLE'

  def find_nearest(self, array: list, value: float):
    array = asarray(array)
    idx = (abs(array - value)).argmin()
    return idx

  def branch(self, model: Model, values_solution):
    var_branch = values_solution[self.find_nearest([i.x for i in values_solution], 0.5)]

    model_0 = model.copy()
    model_0 += var_branch == 0

    model_1 = model.copy()
    model_1 += var_branch == 1

    return (model_0, model_1)

  def print_result(self, model_solved):
    print("Variables:")
    for i in model_solved["variables"]:
      print(i.name, ' = ', i.x)
    print("Objective function:")
    print('Z = ', model_solved["objective"])

  def run(self, filePath):
    variables, rests, coef_obj, coef_rests = self.read_txt(filePath)

    model = Model(sense=MAXIMIZE)

    x = [model.add_var(var_type="CONTINUOUS",
                        lb=0, ub=1, name="x_" + str(i)) for i in range(variables)]

    model.objective = xsum(coef_obj[i]*x[i]
                             for i in range(variables))

    for i in range(rests):
        model += xsum(coef_rests[i][j]*x[j] for j in range(variables)) <= coef_rests[i][-1]

    self.branch_and_bound(model)

    self.print_result(self.solver(self.optimal_model))
