import { registerMethod } from '../registry'
import { Overview }      from './Overview'
import { Description }   from './Description'
import { Formulas }      from './Formulas'
import { Dataset }       from './Dataset'
import { Visualization } from './Visualization'
import { Insights }      from './Insights'

registerMethod({
  id:    'mlp-playground',
  name:  'Neural Network Playground',
  group: 'Neural Networks',
  blurb: 'From-scratch MLP: configurable depth/width, ReLU/Tanh/LeakyReLU/Sigmoid, Adam/RMSProp, L2 + dropout, live decision-boundary heatmap + per-neuron activation maps.',
  Overview,
  Description,
  Formulas,
  Dataset,
  Visualization,
  Insights,
})
