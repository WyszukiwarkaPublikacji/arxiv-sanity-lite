'use strict';


const Image = props => {
    const img = props.image;
    const thumb_img = img.path === '' ? null : <div class='rel_img'><img src={img.path} /></div>;

    return (
    <div class='rel_image'>
        <div class="rel_score">{img.weight.toFixed(2)}</div>
        <div class='rel_id'><a href={'http://arxiv.org/abs/' + img.id}>{img.id}</a></div>
        {thumb_img}
        <div class="rel_abs">{img.caption}</div>
    </div>
    )
}


const ImageList = props => {
    const lst = props.images;
    const ilst = lst.map((jimage, ix) => <Image key={ix} image={jimage} />);
    return (
        <div>
            <div id="imageList" class="rel_images">
                {ilst}
            </div>
        </div>
    )
}


// render images into #wrap
ReactDOM.render(<ImageList images={images} />, document.getElementById('wrap'));
