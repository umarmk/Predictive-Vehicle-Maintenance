import axios from 'axios';

const client = axios.create();

export function predictVehicle(data) {
  return client.post('/predict', data).then((res) => res.data);
}

export function predictTimeseries(data) {
  return client.post('/predict/timeseries', data).then((res) => res.data);
}

export function explain(data) {
  return client.post('/explain', data).then((res) => res.data);
}

export function getHistory() {
  return client.get('/history').then((res) => res.data);
}
