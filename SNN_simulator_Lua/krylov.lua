local krylov = {}

-- Calculates the nth Krylov subspace using Arnoldi iteration

function krylov.krylov(A, n)
  local v = torch.zeros(n, n)
  v[1] = torch.uniform(n)
  v[1] = v[1] / v[1]:norm()

  local h = torch.zeros(n + 1, n)
print(A:size())
print(v:size())
  --[[for k = 2, n do
    q[k] = torch.mm(mat, q[k - 1]:reshape(mat:size(2),1))

    for j = 1, k - 1 do
      h[j][k - 1] = q[j] * q[k]
      q[k]:csub(h[j][k - 1] * q[j])
    end

    h[k][k - 1] = q[k]:norm()
    q[k]:div(h[k][k - 1])
  end]]

  for j = 1, n do
    local Av = A * v[j]
    local w = Av
print(Av:size())
print(w:size())
print(v[j]:size())
    for i = 1, j do
      h[i][j] = Av * v[i]
      w = w - h[i][j] * v[i]
    end

    h[j+1][j] = w:norm()
    if h[j+1][j] == 0 then
      error('Krylov: h -> 0')
    end

    v[j+1] = w / h[j+1][j]
  end

  return q
end

function krylov.pseudoInverse(A)
  return torch.inverse(A * A:t()) * A:t()
end

return krylov
