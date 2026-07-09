from app.services.preprocessing import extract_url_features

if __name__ == '__main__':
    none_meta = extract_url_features(None)
    ex_meta = extract_url_features('https://example.com/news')
    print('None -> vector len', len(none_meta['vector']))
    print('Example -> vector len', len(ex_meta['vector']))
    print('Flags None:', none_meta['flags'])
    print('Domain example:', ex_meta['domain'])
