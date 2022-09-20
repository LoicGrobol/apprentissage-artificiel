module Jekyll
    class NotebookBadges < Liquid::Tag
     def initialize(tag_name, notebook_path, tokens)
        super
        @notebook_path = notebook_path
     end
  
     def render(context)
        return (
            "[![Launch in Binder badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/#{context['site']['repository']}/#{context['site']['repo_branch']}?urlpath=tree/#{@notebook_path})"
        )
     end
    end
  end
  
  Liquid::Template.register_tag('notebook_badges', Jekyll::NotebookBadges)