import { registerMethod } from '../registry'
import { Overview }      from './Overview'
import { Description }   from './Description'
import { Formulas }      from './Formulas'
import { Dataset }       from './Dataset'
import { Visualization } from './Visualization'
import { Insights }      from './Insights'

registerMethod({
  id:    'cnn-visualizer',
  name:  'CNN Visualizer',
  group: 'Deep Learning',
  blurb: 'Animated convolution sweep: watch a kernel slide over an image, writing the feature map cell-by-cell. Conv → ReLU → MaxPool pipeline.',
  Overview,
  Description,
  Formulas,
  Dataset,
  Visualization,
  Insights,
})
