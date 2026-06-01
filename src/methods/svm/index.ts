import { registerMethod } from '../registry'
import { Overview }      from './Overview'
import { Description }   from './Description'
import { Formulas }      from './Formulas'
import { Dataset }       from './Dataset'
import { Visualization } from './Visualization'
import { Insights }      from './Insights'

registerMethod({
  id:    'svm',
  name:  'Support Vector Machine',
  group: 'Classification',
  blurb: 'Kernel soft-margin SVM (linear/poly/RBF) trained via Simplified SMO: animated decision surface, margin bands, support-vector highlighting.',
  Overview,
  Description,
  Formulas,
  Dataset,
  Visualization,
  Insights,
})
