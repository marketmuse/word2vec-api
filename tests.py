import requests
import unittest
import json


# http://34.198.154.215/word2vec/similarity_batch?main_keyword=hubspot&keywords=michael%20johnson%20-photography&keywords=mitchell%20-photography&keywords=digital%20-%20photography%20-%20school&keywords=photography%20-%20lighting&keywords=candid%20-%20photography&keywords=target%20-%20baby%20registry&keywords=baby-%20r-us&keywords=baby%20-cats&keywords=baby%20shower%20-%20games&keywords=babys%20-%20r-us&keywords=seattle%20area%20massage%20&%20wellness%20clinics&keywords=robert%20smith%20-photography&keywords=michael%20johnson%20-photography&keywords=photograph%20-%20lyrics&keywords=mitchell%20-photography&keywords=digital%20-%20photography%20-%20school&keywords=nickelback%20-%20photograph&keywords=def%20leppard%20-%20photograph&keywords=photography%20-%20lighting&keywords=photograph%20-%20nickelback&keywords=pink%20-%20family%20portrait&keywords=family%20portrait%20-%20pink&keywords=pink%20-%20family%20portrait.%20-%20mp3&keywords=pink%20-%20family%20portrait%20mp3&keywords=dog%20food&keywords=red%20bull%20-%20ingredients&keywords=spam%20-%20ingredients&keywords=n%20-nitrosodiethanolamine&keywords=colitis%20---diet&keywords=mediterranean%20-style%20diet&keywords=n%20n%20-diethyl-meta-toluamide&keywords=low%20-%20carb%20diet&keywords=gout%20-%20diet&keywords=anti-%20inflammatory%20diet&keywords=anti%20-%20inflammatory%20diet&keywords=3%20-%20day%20diet&keywords=diet%20-world&keywords=low%20-%20cholesterol%20-%20diet&keywords=kibble%20&%20prentice&keywords=haines%20&%20kibblehouse&keywords=kibbles%20&%20bits&keywords=asset%20based%20lending&keywords=asset%20based%20lending&keywords=san%20diego%20-%20real%20estate&keywords=united%20country%20real%20estate%20-wv&keywords=futon%20-company&keywords=np%20-/it/sistemi-di-sicurezza/company-profile&keywords=new%20york%20&%20company&keywords=bds%20-company&keywords=company%20of%20heroes%20-dev&keywords=coca%20-%20cola%20company&keywords=noodles%20&%20company&keywords=citizens%20bank%20-pa&keywords=bank%20of%20america%20-%20online%20banking&keywords=bank%20of%20america%20-%20online&keywords=coldwell%20-%20banker&keywords=bank%20of%20america%20-%20locations&keywords=chase%20bank%20-%20locations&keywords=free%20-%20credit%20-%20report&keywords=free%20-%20credit%20-%20score&keywords=wright%20-%20patt%20credit%20union&keywords=instagram%20marketing&keywords=aces%20-%20limited%20brands&keywords=marilyn%20manson%20-%20target%20audience&keywords=target%20audience%20-%20marilyn%20manson&keywords=photoshop%20-%20free&keywords=photos-&keywords=free%20stock%20photos%20-royalty&keywords=photosynthesis%20-%20equation&keywords=photoshop%20-%20free%20download&keywords=walmart%20-%20photos&keywords=free%20photos%20-royalty&keywords=roof%20restoration&keywords=duro%20-%20last%20roofing&keywords=medical%20-shingles&keywords=shingles%20-%20symptoms&keywords=shingles%20-%20contagious&keywords=shingles%20-%20roof&keywords=shingles%20-%20disease&keywords=self%20portraits%20-photo&keywords=robert%20smith%20-photography&keywords=michael%20johnson%20-photography&keywords=mitchell%20-photography&keywords=digital%20-%20photography%20-%20school&keywords=photography%20-%20lighting&keywords=candid%20-%20photography&keywords=target%20-%20baby%20registry&keywords=baby-%20r-us&keywords=baby%20-cats&keywords=baby%20shower%20-%20games&keywords=babys%20-%20r-us&keywords=seattle%20area%20massage%20&%20wellness%20clinics&keywords=robert%20smith%20-photography&keywords=michael%20johnson%20-photography&keywords=photograph%20-%20lyrics&keywords=mitchell%20-photography&keywords=digital%20-%20photography%20-%20school&keywords=nickelback%20-%20photograph&keywords=def%20leppard%20-%20photograph&keywords=photography%20-%20lighting&keywords=photograph%20-%20nickelback&keywords=pink%20-%20family%20portrait&keywords=family%20portrait%20-%20pink&keywords=pink%20-%20family%20portrait.%20-%20mp3&keywords=pink%20-%20family%20portrait%20mp3

#URL = 'http://34.198.154.215/word2vec/similarity_batch'  #production
URL = 'http://34.204.210.81/word2vec/similarity_batch'  #development


keywords_500 = ['sql manual', 'how to make a introduction', 'career moms', 'facebook post', 'pictured photography', 'little ways', 'call attribution', 'challenges facing management of organizations today', 'best twitter wallpapers', 'website components', 'word puns', 'repeat visitors', 'facebook log in button', 'powerpoint presentation skill', 'sales pitches', 'conferences in', 'membrain pricing', 'your trade show', 'types of files', 'handy devices', 'what is a tiff file format used for', 'spam free xxx', 'sales analysis template', 'electronic newsletter template', 'crm packages', 'blogger software free', 'bloggers marketing', 'email address searches', 'hiring blogger', 'find email addresses by name', 'domains mail', '100000 views on youtube', 'sales how to', 'developing a digital strategy', 'social media usage', 'directory listing service', 'the future of social networking', 'catchy', 'boomerang sales', 'lean labs', 'how to find keywords on a website', 'child care marketing', 'demographic research tools', 'sales conversion techniques', 'what does crm mean', 'finish line sales', 'google listed', 'business landscape definition', 'teach powerpoint', 'product jingles', 'buttons websites', 'help psychological', 'google pr rank', 'simple homepage', 'seu blog', 'call low', 'facebook share', 'seo web', 'marketing title', 'apps for videos', 'video upload host', 'templates design', 'prospecting plan', 'other social networks', 'marketing lifecycle', 'animated cartoon gifs', 'best coach speeches', 'prospective student', 'small brand', 'list templates', 'right time', 'facebook like button png', 'seo performance', 'goals download', 'website redesign', 'long tail queries', 'backup mobile data', 'statement of work', 'b2b firm', 'get found local', 'mobile company website', 'network cheat', 'download writing', 'marketing class projects', 'webinar setup', 'reese food products', 'how the internet works', 'steps to branding your business', 'inside sales business plan', 'usability standards', 'miller heiman wiki', 'integrate with your', 'funny vacation messages', 'social media marketing services', 'seo link directory', 'telecommunications sales leads', 'create google place page', 'group activities', 'take pictures', 'personal biography', 'competitor insight', 'it company sites', 'how does facebook advertising work', 'reasons program', 'manufacturing benchmarks', 'cro testing', 'marketing a college', 'company jingles', 'marketing a university', 'facebook url', 'pandora box', 'premium content', 'copy paste html codes', 'corporate training quotes', 'business challenges', 'salesforce plugin', 'ppc ad tracking', 'plastic printing', 'company certifications', 'business week rss', 'social media crm software', 'valentine promotion', 'who designed facebook website', 'seo plan', 'powerpoint advice', 'marketing to', 'working tools', 'create an app for my business', 'seo best tools', 'box events', 'email export', 'eventbrite email', 'work the room', 'web design freeware', 'add video to blog', 'sales closing', 'free image post', 'customer story', 'manage deals', 'marketing for manufacturing companies', 'amazon prime case study', 'plus google', 'pictures of twitter logo', 'selling personality', 'customised or customized', 'homepage optimization', 'how to email big attachments', 'software s video', 'thumbnail photos', 'setting up webinars', 'how to get free facebook likes', 'chrome keyboard', 'acquia pricing', 'template for resumes', 'google website for business', 'personalized campaign items', 'organization tools', 'john roberts', 'seo tutorials for beginners', 'resonates', 'dan zarrella', 'massive emails', 'open house forms', 'mobile user', 'marketing director', 'free advertising opportunities', 'saas companies in the bay area', 'seo keywords analyzer', 'sales executive wikipedia', 'book cover ideas', 'app marketplaces', 'usability consultant', 'utm parameters', 'san serif', 'what is a subdomain', 'how to make a perfect resume for job', 'picture phone software', 'ideal personality', 'how can i create blog', 'how to design products', 'entrepreneur blog sites', 'photoshop keyboard', 'clicked', 'image white', 'retail sales technique', 'ranking report tool', 'market hub', 'previous emails', 'pivot table tutorial video', 'verizon iphone', 'google ad ranking', 'human behaviours', 'advertising icons', 'your crm', 'business article submission', 'local business google maps', 'lead quality', 'bad emails', 'make checklist', 'support team', 'duct tape', 'how to calculate sample size', 'make microsoft', 'crm careers', 'sales leads free trial', 'getting jobs', 'advertise to businesses', 'social media channels', 'inbound calls', 'shrink a url', 'marketing a website launch', 'email campaign', 'facebook link widget', 'responsive websites', 'advertising in facebook', 'automation guide', 'multiple mailbox', 'adobe photoshop animation', 'world currency notes', 'my leads', 'script vs screenplay', 'top business blog', 'weed lawn care', 'back to school promotion ideas', 'content marketing blueprint', 'share website', 'business document sharing', 'google seo algorithm', 'comprehensive analysis', 'adword finder', 'database marketing best practices', 'ad format', 'blog page', 'google ad tracker', 'increase clicks', 'tier pricing', 'google search articles', 'power of storytelling', 'help for bloggers', 'corporation organization chart', 'duties of a marketing officer', 'block my ip', 'mobile ad banner', 'buyer behavior', 'email bounce codes', 'salesforce offline', 'rich mails', 'free fun fonts', 'personality type a or b test', 'catchy title examples', 'website analysis template', 'customer ratings', 'channel marketing', 'sales executives jobs', 'banner rates', 'email marketing stats', 'search engine optimization key words', 'social tweet', 'purple red', 'motivated speech', 'salesforce forums', 'duplicate content', 'brightedge blog', '301 redirect aspx', 'hotel marketing report', 'e mail notifications', 'apple mac pc', 'software more', 'seo xls', 'ecommerce shopping basket', 'what are blogs', 'collecting design', 'ad words bid', 'setting up a shopping cart', 'language puns', 'identity product', 'subliminal marketing techniques', 'networks advertising', 'effective way', 'facebook for business tutorial', 'how to edit website', 'direct response tv commercials', 'sales graphs', 'cheapest stock photos', 'ad agency rfp', 'creative emails', 'how to make your valentine day special', 'find person', 'leo and burnett', 'html anchor links', 'free isp list', 'best agency websites', 'sales advertisements', 'type media', 'best deal software', 'selling advice', 'how to be more assertive', 'content marketing ebook', 'branding essentials', 'page likes on facebook', 'questions survey', 'password protect pages', 'workflow api', 'html coding email', 'website to advertise', 'create a web address', 'big diamond wedding rings', 'create new company', 'facebook layout design', 'outlook feedback', 'how to find keyword popularity', 'sales rep for hair products', 'video blog', 'free software project management tool', 'podcasting blog', 'convert graphics', '7 elements of design', 'get keyword', 'how to make account on google', 'signature analysis', 'e newsletters', 'site analysis', 'image resized', 'alt search', 'seo links page', 'closed loops', 'facebook in an iframe', 'loyalty values', 'critiquing websites', 'blogging for your business', 'github monitoring', 'subdomain godaddy', 'david mihm local search', 'sales funnel definition', 'seo efforts', 'integrate tool', 'fundraising campaign', 'selling with insights', 'best daily websites', 'visual content creator', 'hudson horizon', 'how to write bio for website', 'innovative ideas', 'creative strategy brief', 'live sales', 'multiple inbox', 'task reminder', 'buy marketing leads', 'slideshare', 'survey project', 'top google extensions', 'workflow examples', 'pay per click advertising ppc', 'mind relief', 'omnichannel customer experience', 'role', 'effective managers', 'ssae 16 professionals', 'addresses listings', 'twitter templates', 'outlook close', 'average cost of a website', 'webmaster tool', 'twitter linkedin', 'graphic templates', 'what is css', 'table excel function', 'what is sales and marketing', 'marketing a new brand', 'facebook page wall settings', 'code template', 'salespeople with a purpose', 'what is a push notification', 'twitter ppc', 'public relations roi', 'paid ad', 'emails sender', 'job research', 'process agency', 'best thumbnail size', 'positive language', 'what is the conversion rate', 'youtube search optimization', 'make form', 'hispanic marketing companies', 'good free blog', 'use excel', 'the history of advertising', 'best web designer in the world', 'motivate sales teams', 'calling numbers', 'crm dynamics microsoft', 'twitter like buttons', 'random thought', 'apple fans', 'facebook campaign', 'advertising industry jobs', 'sample territory sales plan', 'creating advertising', 'seamless customer experiences', 'google add campaign', 'streaming application', 'new resolutions', 'image as a link', 'business statements', 'open source task list software', 'adwords spy ppc', 'ad tips', 'email marketing promotion', 'free keyword report', 'online tour operator', 'top ecommerce designs', 'retention tool', 'how style', 'drip email campaign', 'smart forms', 'contact price', 'spam mail list', 'i log', 'price alerts', 'email subject link', 'timeline picture size', 'validate email addresses', 'free emails services', 'marketing specialist', 'email marketing segmentation', 'inbound call management', 'email marketing softwares', 'conversion works', 'customer service representative', 'business effective', 'strategy video', 'the coolest websites', 'free video web sites', 'pr monitoring', 'crm macintosh', 'blog script', 'presentation graph', 'edit photos for facebook free', 'web page quotes', 'twitter widgets website', 'industry expert', 'find information on anyone', 'online community management software', 'create template', 'twitter links', 'sydney career', 'encrypting website', 'locating email addresses', 'get fans', 'local business information', 'permission marketing', 'download corporate fonts', 'great place to work criteria', 'examples of queries', 'ppc tips', 'photograph mobile', 'hispanic marketing advertising', 'how do i create a powerpoint template', 'physician email database', 'photoshop cs keyboard shortcuts', 'sleep happens', 'facebook planning', 'free blog creators', 'estimating a project', 'writing search engine', 'branding partner', 'branding for startup', 'desktop clutter', 'linkedin events', 'rate this website', 'find emails online', 'disconnect', 'how to research a company', 'hiring writers', 'brainstorm session', 'marketing university', 'sales support job titles', 'facebook ad picture size', 'event invites', 'sales today', 'action email', 'social publishing websites', 'qa team', 'email marketing crm software', 'download computer fonts', 'your hub', 'personalizing', 'modern businesses', 'sale forecasting', 'single contact', 'para melhorar', 'job interviewer', 'create layout', 'publishing blog', 'publishers marketing', 'put a business on google maps', 'popup websites', 'automated crm', 'magnets for you']
keywords_500_fails = ['michael johnson -photography', 'mitchell -photography', 'digital - photography - school', 'photography - lighting', 'candid - photography', 'target - baby registry', 'baby- r-us', 'baby -cats', 'baby shower - games', 'babys - r-us', 'seattle area massage & wellness clinics', 'robert smith -photography', 'michael johnson -photography', 'photograph - lyrics', 'mitchell -photography', 'digital - photography - school', 'nickelback - photograph', 'def leppard - photograph', 'photography - lighting', 'photograph - nickelback', 'pink - family portrait', 'family portrait - pink', 'pink - family portrait. - mp3', 'pink - family portrait mp3', 'dog food', 'red bull - ingredients', 'spam - ingredients', 'n -nitrosodiethanolamine', 'colitis ---diet', 'mediterranean -style diet', 'n n -diethyl-meta-toluamide', 'low - carb diet', 'gout - diet', 'anti- inflammatory diet', 'anti - inflammatory diet', '3 - day diet', 'diet -world', 'low - cholesterol - diet', 'kibble & prentice', 'haines & kibblehouse', 'kibbles & bits', 'asset based lending', 'asset based lending', 'san diego - real estate', 'united country real estate -wv', 'futon -company', 'np -/it/sistemi-di-sicurezza/company-profile', 'new york & company', 'bds -company', 'company of heroes -dev', 'coca - cola company', 'noodles & company', 'citizens bank -pa', 'bank of america - online banking', 'bank of america - online', 'coldwell - banker', 'bank of america - locations', 'chase bank - locations', 'free - credit - report', 'free - credit - score', 'wright - patt credit union', 'instagram marketing', 'aces - limited brands', 'marilyn manson - target audience', 'target audience - marilyn manson', 'photoshop - free', 'photos', 'free stock photos -royalty', 'photosynthesis - equation', 'photoshop - free download', 'walmart - photos', 'free photos -royalty', 'roof restoration', 'duro - last roofing', 'medical -shingles', 'shingles - symptoms', 'shingles - contagious', 'shingles - roof', 'shingles - disease', 'self portraits -photo', 'robert smith -photography', 'michael johnson -photography', 'mitchell -photography', 'digital - photography - school', 'photography - lighting', 'candid - photography', 'target - baby registry', 'baby- r-us', 'baby -cats', 'baby shower - games', 'babys - r-us', 'seattle area massage & wellness clinics', 'robert smith -photography', 'michael johnson -photography', 'photograph - lyrics', 'mitchell -photography', 'digital - photography - school', 'nickelback - photograph', 'def leppard - photograph', 'photography - lighting', 'photograph - nickelback', 'pink - family portrait', 'family portrait - pink', 'pink - family portrait. - mp3', 'pink - family portrait mp3', 'dog food', 'red bull - ingredients', 'spam - ingredients', 'n -nitrosodiethanolamine', 'colitis ---diet', 'mediterranean -style diet', 'n n -diethyl-meta-toluamide', 'low - carb diet', 'gout - diet', 'anti- inflammatory diet', 'anti - inflammatory diet', '3 - day diet', 'diet -world', 'low - cholesterol - diet', 'kibble & prentice', 'haines & kibblehouse', 'kibbles & bits', 'asset based lending', 'asset based lending', 'san diego - real estate', 'united country real estate -wv', 'futon -company', 'np -/it/sistemi-di-sicurezza/company-profile', 'new york & company', 'bds -company', 'company of heroes -dev', 'coca - cola company', 'noodles & company', 'citizens bank -pa', 'bank of america - online banking', 'bank of america - online', 'coldwell - banker', 'bank of america - locations', 'chase bank - locations', 'free - credit - report', 'free - credit - score', 'wright - patt credit union', 'instagram marketing', 'aces - limited brands', 'marilyn manson - target audience', 'target audience - marilyn manson', 'photoshop - free', 'photos', 'free stock photos -royalty', 'photosynthesis - equation', 'photoshop - free download', 'walmart - photos', 'free photos -royalty', 'roof restoration', 'duro - last roofing', 'medical -shingles', 'shingles - symptoms', 'shingles - contagious', 'shingles - roof', 'shingles - disease', 'self portraits -photo', 'robert smith -photography', 'michael johnson -photography', 'mitchell -photography', 'digital - photography - school', 'photography - lighting', 'candid - photography', 'target - baby registry', 'baby- r-us', 'baby -cats', 'baby shower - games', 'babys - r-us', 'seattle area massage & wellness clinics', 'robert smith -photography', 'michael johnson -photography', 'photograph - lyrics', 'mitchell -photography', 'digital - photography - school', 'nickelback - photograph', 'def leppard - photograph', 'photography - lighting', 'photograph - nickelback', 'pink - family portrait', 'family portrait - pink', 'pink - family portrait. - mp3', 'pink - family portrait mp3', 'dog food', 'red bull - ingredients', 'spam - ingredients', 'n -nitrosodiethanolamine', 'colitis ---diet', 'mediterranean -style diet', 'n n -diethyl-meta-toluamide', 'low - carb diet', 'gout - diet', 'anti- inflammatory diet', 'anti - inflammatory diet', '3 - day diet', 'diet -world', 'low - cholesterol - diet', 'kibble & prentice', 'haines & kibblehouse', 'kibbles & bits', 'asset based lending', 'asset based lending', 'san diego - real estate', 'united country real estate -wv', 'futon -company', 'np -/it/sistemi-di-sicurezza/company-profile', 'new york & company', 'bds -company', 'company of heroes -dev', 'coca - cola company', 'noodles & company', 'citizens bank -pa', 'bank of america - online banking', 'bank of america - online', 'coldwell - banker', 'bank of america - locations', 'chase bank - locations', 'free - credit - report', 'free - credit - score', 'wright - patt credit union', 'instagram marketing', 'aces - limited brands', 'marilyn manson - target audience', 'target audience - marilyn manson', 'photoshop - free', 'photos', 'free stock photos -royalty', 'photosynthesis - equation', 'photoshop - free download', 'walmart - photos', 'free photos -royalty', 'roof restoration', 'duro - last roofing', 'medical -shingles', 'shingles - symptoms', 'shingles - contagious', 'shingles - roof', 'shingles - disease', 'self portraits -photo', 'robert smith -photography', 'michael johnson -photography', 'mitchell -photography', 'digital - photography - school', 'photography - lighting', 'candid - photography', 'target - baby registry', 'baby- r-us', 'baby -cats', 'baby shower - games', 'babys - r-us', 'seattle area massage & wellness clinics', 'robert smith -photography', 'michael johnson -photography', 'photograph - lyrics', 'mitchell -photography', 'digital - photography - school', 'nickelback - photograph', 'def leppard - photograph', 'photography - lighting', 'photograph - nickelback', 'pink - family portrait', 'family portrait - pink', 'pink - family portrait. - mp3', 'pink - family portrait mp3', 'dog food', 'red bull - ingredients', 'spam - ingredients', 'n -nitrosodiethanolamine', 'colitis ---diet', 'mediterranean -style diet', 'n n -diethyl-meta-toluamide', 'low - carb diet', 'gout - diet', 'anti- inflammatory diet', 'anti - inflammatory diet', '3 - day diet', 'diet -world', 'low - cholesterol - diet', 'kibble & prentice', 'haines & kibblehouse', 'kibbles & bits', 'asset based lending', 'asset based lending', 'san diego - real estate', 'united country real estate -wv', 'futon -company', 'np -/it/sistemi-di-sicurezza/company-profile', 'new york & company', 'bds -company', 'company of heroes -dev', 'coca - cola company', 'noodles & company', 'citizens bank -pa', 'bank of america - online banking', 'bank of america - online', 'coldwell - banker', 'bank of america - locations', 'chase bank - locations', 'free - credit - report', 'free - credit - score', 'wright - patt credit union', 'instagram marketing', 'aces - limited brands', 'marilyn manson - target audience', 'target audience - marilyn manson', 'photoshop - free', 'photos', 'free stock photos -royalty', 'photosynthesis - equation', 'photoshop - free download', 'walmart - photos', 'free photos -royalty', 'roof restoration', 'duro - last roofing', 'medical -shingles', 'shingles - symptoms', 'shingles - contagious', 'shingles - roof', 'shingles - disease', 'self portraits -photo', 'robert smith -photography', 'michael johnson -photography', 'mitchell -photography', 'digital - photography - school', 'photography - lighting', 'candid - photography', 'target - baby registry', 'baby- r-us', 'baby -cats', 'baby shower - games', 'babys - r-us', 'seattle area massage & wellness clinics', 'robert smith -photography', 'michael johnson -photography', 'photograph - lyrics', 'mitchell -photography', 'digital - photography - school', 'nickelback - photograph', 'def leppard - photograph', 'photography - lighting', 'photograph - nickelback', 'pink - family portrait', 'family portrait - pink', 'pink - family portrait. - mp3', 'pink - family portrait mp3', 'dog food', 'red bull - ingredients', 'spam - ingredients', 'n -nitrosodiethanolamine', 'colitis ---diet', 'mediterranean -style diet', 'n n -diethyl-meta-toluamide', 'low - carb diet', 'gout - diet', 'anti- inflammatory diet', 'anti - inflammatory diet', '3 - day diet', 'diet -world', 'low - cholesterol - diet', 'kibble & prentice', 'haines & kibblehouse', 'kibbles & bits', 'asset based lending', 'asset based lending', 'san diego - real estate', 'united country real estate -wv', 'futon -company', 'np -/it/sistemi-di-sicurezza/company-profile', 'new york & company', 'bds -company', 'company of heroes -dev', 'coca - cola company', 'noodles & company', 'citizens bank -pa', 'bank of america - online banking', 'bank of america - online', 'coldwell - banker', 'bank of america - locations', 'chase bank - locations', 'free - credit - report', 'free - credit - score', 'wright - patt credit union', 'instagram marketing', 'aces - limited brands', 'marilyn manson - target audience', 'target audience - marilyn manson', 'photoshop - free', 'photos', 'free stock photos -royalty', 'photosynthesis - equation', 'photoshop - free download', 'walmart - photos', 'free photos -royalty', 'roof restoration', 'duro - last roofing', 'medical -shingles', 'shingles - symptoms', 'shingles - contagious', 'shingles - roof', 'shingles - disease', 'self portraits -photo', 'robert smith -photography', 'michael johnson -photography', 'mitchell -photography', 'digital - photography - school', 'photography - lighting', 'candid - photography', 'target - baby registry', 'baby- r-us', 'baby -cats', 'baby shower - games', 'babys - r-us', 'seattle area massage & wellness clinics', 'robert smith -photography', 'michael johnson -photography', 'photograph - lyrics', 'mitchell -photography', 'digital - photography - school', 'nickelback - photograph', 'def leppard - photograph', 'photography - lighting', 'photograph - nickelback', 'pink - family portrait', 'family portrait - pink', 'pink - family portrait. - mp3', 'pink - family portrait mp3', 'dog food', 'red bull - ingredients', 'spam - ingredients', 'n -nitrosodiethanolamine', 'colitis ---diet', 'mediterranean -style diet', 'n n -diethyl-meta-toluamide', 'low - carb diet', 'gout - diet', 'anti- inflammatory diet', 'anti - inflammatory diet', '3 - day diet', 'diet -world', 'low - cholesterol - diet', 'kibble & prentice', 'haines & kibblehouse', 'kibbles & bits', 'asset based lending', 'asset based lending', 'san diego - real estate', 'united country real estate -wv', 'futon -company', 'np -/it/sistemi-di-sicurezza/company-profile', 'new york & company', 'bds -company', 'company of heroes -dev', 'coca - cola company', 'noodles & company', 'citizens bank -pa', 'bank of america - online banking', 'bank of america - online', 'coldwell - banker', 'bank of america - locations', 'chase bank - locations', 'free - credit - report', 'free - credit - score', 'wright - patt credit union', 'instagram marketing', 'photos', 'free stock photos -royalty', 'free photos -royalty', 'roof restoration', 'medical -shingles', 'self portraits -photo', 'robert smith -photography', 'michael johnson -photography', 'mitchell -photography', 'baby- r-us', 'baby -cats', 'seattle area massage & wellness clinics', 'robert smith -photography', 'michael johnson -photography', 'mitchell -photography', 'dog food', 'n -nitrosodiethanolamine', 'colitis ---diet', 'mediterranean -style diet', 'n n -diethyl-meta-toluamide', 'anti- inflammatory diet', 'diet -world', 'kibble & prentice', 'haines & kibblehouse', 'kibbles & bits', 'asset based lending', 'asset based lending', 'united country real estate -wv', 'futon -company', 'np -/it/sistemi-di-sicurezza/company-profile', 'new york & company', 'bds -company', 'company of heroes -dev', 'noodles & company', 'citizens bank -pa', 'instagram marketing', 'aces - limited brands', 'marilyn manson - target audience', 'target audience - marilyn manson', 'photoshop - free', 'photos', 'free stock photos -royalty', 'photosynthesis - equation', 'photoshop - free download', 'walmart - photos', 'free photos -royalty', 'roof restoration', 'duro - last roofing', 'medical -shingles', 'shingles - symptoms', 'shingles - contagious', 'shingles - roof', 'shingles - disease', 'self portraits -photo', 'robert smith -photography', 'michael johnson -photography', 'mitchell -photography', 'digital - photography - school', 'photography - lighting', 'candid - photography', 'target - baby registry', 'baby- r-us', 'baby -cats', 'baby shower - games', 'babys - r-us', 'seattle area massage & wellness clinics', 'robert smith -photography', 'michael johnson -photography', 'photograph - lyrics', 'mitchell -photography', 'digital - photography - school', 'nickelback - photograph', 'def leppard - photograph', 'photography - lighting', 'photograph - nickelback', 'pink - family portrait', 'family portrait - pink', 'pink - family portrait. - mp3', 'pink - family portrait mp3', 'dog food', 'red bull - ingredients', 'spam - ingredients', 'n -nitrosodiethanolamine', 'colitis ---diet', 'mediterranean -style diet', 'n n -diethyl-meta-toluamide', 'low - carb diet', 'gout - diet', 'anti- inflammatory diet', 'anti - inflammatory diet', '3 - day diet', 'diet -world', 'low - cholesterol - diet', 'kibble & prentice', 'haines & kibblehouse', 'kibbles & bits', 'asset based lending', 'asset based lending', 'san diego - real estate', 'united country real estate -wv', 'futon -company', 'np -/it/sistemi-di-sicurezza/company-profile', 'new york & company', 'bds -company', 'company of heroes -dev', 'coca - cola company', 'noodles & company', 'citizens bank -pa', 'bank of america - online banking', 'bank of america - online', 'coldwell - banker', 'bank of america - locations', 'chase bank - locations', 'free - credit - report', 'free - credit - score', 'wright - patt credit union', 'instagram marketing', 'aces - limited brands', 'marilyn manson - target audience', 'target audience - marilyn manson', 'photoshop - free', 'photos', 'free stock photos -royalty', 'photosynthesis - equation', 'photoshop - free download', 'walmart - photos', 'free photos -royalty', 'roof restoration', 'duro - last roofing', 'medical -shingles', 'shingles - symptoms', 'shingles - contagious', 'shingles - roof', 'shingles - disease', 'self portraits -photo', 'robert smith -photography', 'michael johnson -photography', 'mitchell -photography', 'digital - photography - school', 'photography - lighting', 'candid - photography', 'target - baby registry', 'baby- r-us', 'baby -cats', 'baby shower - games', 'babys - r-us', 'seattle area massage & wellness clinics', 'robert smith -photography', 'michael johnson -photography', 'photograph - lyrics', 'mitchell -photography', 'digital - photography - school', 'nickelback - photograph', 'def leppard - photograph', 'photography - lighting', 'photograph - nickelback', 'pink - family portrait', 'family portrait - pink', 'pink - family portrait. - mp3', 'pink - family portrait mp3', 'dog food', 'red bull - ingredients', 'spam - ingredients', 'n -nitrosodiethanolamine', 'colitis ---diet', 'mediterranean -style diet', 'n n -diethyl-meta-toluamide', 'low - carb diet', 'gout - diet', 'anti- inflammatory diet', 'anti - inflammatory diet', '3 - day diet', 'diet -world', 'low - cholesterol - diet', 'kibble & prentice', 'haines & kibblehouse', 'kibbles & bits', 'asset based lending', 'asset based lending', 'san diego - real estate', 'united country real estate -wv', 'futon -company', 'np -/it/sistemi-di-sicurezza/company-profile', 'new york & company', 'bds -company', 'company of heroes -dev', 'coca - cola company', 'noodles & company', 'citizens bank -pa', 'bank of america - online banking', 'bank of america - online', 'coldwell - banker', 'bank of america - locations', 'chase bank - locations', 'free - credit - report', 'free - credit - score', 'wright - patt credit union']
keywords_50 = ['michael johnson -photography', 'mitchell -photography', 'digital - photography - school', 'photography - lighting', 'candid - photography', 'target - baby registry', 'baby- r-us', 'baby -cats', 'baby shower - games', 'babys - r-us', 'seattle area massage & wellness clinics', 'robert smith -photography', 'michael johnson -photography', 'photograph - lyrics', 'mitchell -photography', 'digital - photography - school', 'nickelback - photograph', 'def leppard - photograph', 'photography - lighting', 'photograph - nickelback', 'pink - family portrait', 'family portrait - pink', 'pink - family portrait. - mp3', 'pink - family portrait mp3', 'dog food', 'red bull - ingredients', 'spam - ingredients', 'n -nitrosodiethanolamine', 'colitis ---diet', 'mediterranean -style diet', 'n n -diethyl-meta-toluamide', 'low - carb diet', 'gout - diet', 'anti- inflammatory diet', 'anti - inflammatory diet', '3 - day diet', 'diet -world', 'low - cholesterol - diet', 'kibble & prentice', 'haines & kibblehouse', 'kibbles & bits', 'asset based lending', 'asset based lending', 'san diego - real estate', 'united country real estate -wv', 'futon -company', 'np -/it/sistemi-di-sicurezza/company-profile', 'new york & company', 'bds -company', 'company of heroes -dev']

all_keywords_in_google = ['nitrile', 'garden']

google_only_main_keyword = 'vigoda'
facebook_only_main_keyword = 'youtubing'

google_facebook_main_keyword = 'nitrile'

nor_google_facebook_main_keyword = 'edjskaoexjdjeowowksk'


def keyword_500_fail_test():
  r = requests.post(URL, data= { 'main_keyword': 'hubspot', 'keywords': keywords_500_fails})

  assert r.status_code == 200, "status code should be 200"

  j_res = json.loads(r.text)

  assert j_res['main_keyword'] == 'hubspot', "does not return correct main topic"
  assert len(j_res['fails']) == 4, "does not return correct number of failed words"
  assert len(j_res['semantic_similarity_scores']) == 71, "does not return correct number of successfully compared words"

  return r

def keyword_500_test():
  r = requests.post(URL, data= { 'main_keyword': 'hubspot', 'keywords': keywords_500})

  assert r.status_code == 200, "status code should be 200"

  j_res = json.loads(r.text)

  assert j_res['main_keyword'] == 'hubspot', "does not return correct main topic"
  assert len(j_res['fails']) == 4, "does not return correct number of failed words"
  assert len(j_res['semantic_similarity_scores']) == 496, "does not return correct number of successfully compared words"

  return r



def all_keywords_found_in_google():
  r = requests.post(URL, data= { 'main_keyword': 'hubspot', 'keywords': all_keywords_in_google})

  assert r.status_code == 200, "status code should be 200"

  j_res = json.loads(r.text)

  assert j_res['main_keyword'] == 'hubspot', "does not return correct main topic"
  assert len(j_res['fails']) == 0, "does not return correct number of failed words"
  assert len(j_res['semantic_similarity_scores']) == 2, "does not return correct number of successfully compared words"

  return r


def main_keyword_only_found_in_google():
  r = requests.post(URL, data= { 'main_keyword': google_only_main_keyword, 'keywords': keywords_50})

  assert r.status_code == 200, "status code should be 200"

  j_res = json.loads(r.text)

  assert j_res['main_keyword'] == google_only_main_keyword, "does not return correct main topic"
  assert len(j_res['fails']) == 4, "does not return correct number of failed words"
  assert len(j_res['semantic_similarity_scores']) == 41, "does not return correct number of successfully compared words"

  return r



def main_keyword_only_found_in_facebook():
  r = requests.post(URL, data= { 'main_keyword': facebook_only_main_keyword, 'keywords': keywords_50})

  assert r.status_code == 200, "status code should be 200"

  j_res = json.loads(r.text)

  assert j_res['main_keyword'] == facebook_only_main_keyword, "does not return correct main topic"
  assert len(j_res['fails']) == 7, "does not return correct number of failed words"
  assert len(j_res['semantic_similarity_scores']) == 38, "does not return correct number of successfully compared words"

  return r


def main_keyword_found_in_both():
  r = requests.post(URL, data= { 'main_keyword': google_facebook_main_keyword, 'keywords': keywords_50})

  assert r.status_code == 200, "status code should be 200"

  j_res = json.loads(r.text)

  assert j_res['main_keyword'] == google_facebook_main_keyword, "does not return correct main topic"
  assert len(j_res['fails']) == 4, "does not return correct number of failed words"
  assert len(j_res['semantic_similarity_scores']) == 41, "does not return correct number of successfully compared words"


  return r


def main_keyword_found_in_none():
  r = requests.post(URL, data= { 'main_keyword': nor_google_facebook_main_keyword, 'keywords': keywords_50})

  assert r.status_code == 400, "status code should be 400"

  return r


keyword_500_fail_test()
keyword_500_test()
all_keywords_found_in_google()
main_keyword_only_found_in_google()
main_keyword_only_found_in_facebook()
main_keyword_found_in_both()
main_keyword_found_in_none()