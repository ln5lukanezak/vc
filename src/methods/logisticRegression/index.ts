import { registerMethod } from '../registry'
import { Overview }      from './Overview'
import { Description }   from './Description'
import { Formulas }      from './Formulas'
import { Dataset }       from './Dataset'
import { Visualization } from './Visualization'
import { Insights }      from './Insights'

registerMethod({
  id:    'logistic-regression',
  name:  'Logistic & Softmax Regression',
  group: 'Classification',
  blurb: 'From-scratch softmax regression (K=2–4 classes): cross-entropy + L2, animated probability-surface heatmap, confusion matrix, Adam/SGD/RMSProp/Momentum.',
  Overview,
  Description,
  Formulas,
  Dataset,
  Visualization,
  Insights,
})
