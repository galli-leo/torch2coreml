require 'torch'
require 'nn'
require 'dpnn'

-- require 'fast-neural-style.fast_neural_style.ShaveImage'
-- require 'fast-neural-style.fast_neural_style.TotalVariation'
-- require 'fast-neural-style.fast_neural_style.InstanceNormalization'



local function replaceModule(x, name, create_fn)
  if not x.modules then
    return
  end
  for i = 1,#x.modules do
    m = x.modules[i]
    if m.__typename == name then
      x.modules[i] = create_fn(m)
    end
    replaceModule(m, name, create_fn)
  end
end

local function replaceInception(x)
  if not x.modules then
    return
  end
  for i = 1,#x.modules do
    m = x.modules[i]
    if m.__typename == 'nn.Inception' then
      print(m.module)
      x.modules[i] = m.module
    end
    replaceInception(m)
  end
end

local function main()
  local cmd = torch.CmdLine()
  cmd:option('-input', '')
  cmd:option('-output', '')
  local opt = cmd:parse(arg)
  local model = torch.load(opt.input)

  -- Replace nn.ShaveImage with crop using SpatialZeroPadding with negative offsets
  -- replaceModule(model, 'nn.ShaveImage', function(m)
  --   local size = m.size
  --   return nn.SpatialZeroPadding(-size, -size, -size, -size)
  -- end)
  --
  replaceModule(model, 'nn.SpatialConvolutionMM', function(n)
    torch.setdefaulttensortype('torch.FloatTensor')
    local new = nn.SpatialConvolution(n.nInputPlane, n.nOutputPlane, n.kW, n.kH, n.dW, n.dH, n.padW, n.padH)
    torch.setdefaulttensortype('torch.LongTensor')
    new.weight = n.weight
--    new.weight = new.weight:float()
    new.bias = n.bias
    new.gradWeight = n.gradWeight
    new.gradBias = n.gradBias
--    new.gradWeight = new.gradWeight:float()
--    new.gradBias = new.gradBias:double()
    --new.bias = n.bias
--    new:type(float)
    --print(new.bias)
    --print(torch.Tensor())
    return new
  end)

  replaceInception(model)

  x = torch.FloatTensor()
  model:forward(x:resize(1,3,96,96))
  -- -- Remove last TotalVariation layer
  -- if model.modules[#model.modules].__typename == 'nn.TotalVariation' then
  --   model.modules[#model.modules] = nil
  -- end
  print(model)
  -- Save prepared model
  torch.save(opt.output, model)
end

main()
