module Jekyll
   class NotebookBadges < Liquid::Tag
      def initialize(tag_name, notebook_path, tokens)
         super
         @notebook_path = notebook_path
      end
  
      def render(context)
          if context['site'].key?("environ_repository")
            repo_dir = context['site']['repository'].split("/").last
            urlpath = (
               "?repo=#{ERB::Util.url_encode("https://github.com/" + context['site']['repository'])}" +
               "&urlpath=#{ERB::Util.url_encode("tree/#{repo_dir}/#{@notebook_path}")}" +
               "&branch=#{context['site']['repo_branch']}"
            )
            urlpath_escaped = ERB::Util.url_encode(urlpath)
            res = (
               "[![Launch in Binder badge](https://mybinder.org/badge_logo.svg)]" +
               "(https://mybinder.org/v2/gh/#{context['site']['environ_repository']}/#{context['site']['repo_branch']}" +
               "?urlpath=git-pull#{urlpath_escaped})"
            )
         else
            res = "[![Launch in Binder badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/#{context['site']['repository']}/#{context['site']['repo_branch']}?urlpath=tree/#{@notebook_path})"
         end
         return res
      end
   end
end

module Jekyll
   class NotebookLink < Liquid::Tag
      def initialize(tag_name, notebook_path, tokens)
         super
         @notebook_path = notebook_path
      end
  
      def render(context)
         res = "[![Launch in Binder badge](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/#{context['site']['repository']}/#{context['site']['repo_branch']}?urlpath=tree/#{@notebook_path})"
         return res
      end
   end
end

Liquid::Template.register_tag('notebook_badges', Jekyll::NotebookBadges)