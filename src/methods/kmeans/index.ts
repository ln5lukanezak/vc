import { registerMethod } from '../registry'
import { Overview }      from './Overview'
import { Description }   from './Description'
import { Formulas }      from './Formulas'
import { Dataset }       from './Dataset'
import { Visualization } from './Visualization'
import { Insights }      from './Insights'

registerMethod({
  id:    'kmeans',
  name:  'K-Means Clustering',
  group: 'Unsupervised',
  blurb: 'Lloyd\'s algorithm + K-Means++ init: animated assign/update, centroid trails, Voronoi regions, inertia curve, elbow method.',
  Overview,
  Description,
  Formulas,
  Dataset,
  Visualization,
  Insights,
})
