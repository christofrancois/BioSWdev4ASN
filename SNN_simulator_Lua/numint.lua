-- All numerical integration methods should have the following format:
-- f: derivate function of time and value (f(t,y)) to be approximated
-- t_0: starting time
-- y_0: value at starting time
-- h: time step / step size
-- ...: temporary variables for derivate function

local numint = {}

function numint.euler(f, t_0, y_0, h, ...)
  return y_0 + h * f(t_0, y_0, ...)
end

function numint.RK4(f, t_0, y_0, h, ...)
  local h2 = h/2
  local k1 = f(t_0, y_0, ...)
  local k2 = f(t_0 + h2, y_0 + h2 * k1, ...)
  local k3 = f(t_0 + h2, y_0 + h2 * k2, ...)
  local k4 = f(t_0 + h, y_0 + h * k3, ...)
  return y_0 + (h/6)*(k1 + 2*(k2 + k3) + k4)
end

function numint.adaptive(method, order, tol)
  assert(method, 'numint.adaptive: A method must be given')
  assert(type(order) == 'number'
    , 'numint.adaptive: The order must be given and must be a number!')
  tol = tol or 1e-8 -- Hat constant
  local adaptive
  adaptive = function(f, t_0, y_0, h_0, ...)
    local h_02 = h_0/2
    local x1 = method(f, t_0, y_0, h_0, ...)
    local x2 = method(f, t_0, y_0, h_02, ...)
    x2 = method(f, t_0+h_02, x2, h_02, ...)
    local epsilon = torch.abs(x1 - x2) / (math.pow(2, order) - 1)

    if torch.max(epsilon) > tol then
      local first_half = adaptive(f, t_0, y_0, h_02, ...)
      return adaptive(f, t_0 + h_02, first_half, h_02, ...)
    end

    return x2 + epsilon
  end
  return adaptive
end

return numint
