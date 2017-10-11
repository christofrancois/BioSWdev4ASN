local resmets = {}

-- These methods accept matrices of dimensions n*y*x
-- and y*x, where n is the number of channels.
-- The returned matrix always has three dimensions.

function resmets.nn(mat, w, h)
  if mat:dim() == 2 then
    mat:reshape(1, mat:size(1), mat:size(2))
  end

  local result = torch.zeros(mat:size(1), h, w)
  local oh, ow = mat:size(2), mat:size(3)

  for k = 1, mat:size(1) do
    for i = 1, w do
      for j = 1, h do
        result[k][j][i] = mat[k][math.floor(oh/h*j)][math.floor(ow/w*i)]
      end
    end
  end

  return result
end

return resmets
