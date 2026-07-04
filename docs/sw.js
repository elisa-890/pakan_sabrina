// Service worker minimal - hanya untuk memenuhi syarat PWA installable.
// Tidak melakukan caching agresif supaya aplikasi Streamlit di dalam iframe
// selalu memuat versi terbaru dari server.

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  // Pass-through: biarkan semua request berjalan normal ke jaringan.
  event.respondWith(fetch(event.request));
});
