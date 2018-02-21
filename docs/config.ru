use Rack::Static,
    :urls => ['/stylesheets', '/fonts', '/images', '/includes', '/javascripts', '/layouts'],
    :root => 'docs',
    :index => 'index.html'

run lambda { |env|
  [
    200,
    {
      'Content-Type'  => 'text/html',
      'Cache-Control' => 'public, max-age=86400'
    },
    File.open('docs/index.html', File::RDONLY)
  ]
}
