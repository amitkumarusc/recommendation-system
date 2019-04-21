$("#typeaheader").typeahead({
    hint: true,
    highlight: true,
    minLength: 1
  },
  {
    display: 'title',
    async: true,
    source: function (keyword, processSync, processAsync) {
      return $.get('/search/json', { keyword: keyword }, function (data) {
        return processAsync(data);
      });
    },
    templates: {
      empty: function() {
        return '<div class="EmptyMessage">EMPTY !!!</div>';
      },
      suggestion: function(movie) {
        return '<div class="search-bar-item"><img class="search-bar-item-img" src="'+ movie.url+'"><div class="search-bar-item-title">' + movie.title + '</div></div>';
      }
    }
});