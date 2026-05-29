import { registerMethod } from '../registry'
import { Overview } from './Overview'
import { Description } from './Description'
import { Formulas } from './Formulas'
import { Dataset } from './Dataset'
import { Visualization } from './Visualization'
import { Insights } from './Insights'

registerMethod({
  id: 'linear-regression',
  name: 'Linear & Polynomial Regression',
  group: 'Regression',
  blurb: 'Fit a polynomial curve via gradient descent; MSE loss, live R² readout.',
  Overview,
  Description,
  Formulas,
  Dataset,
  Visualization,
  Insights,
})
