'use client';
import Image from 'next/image';
import React, { useState } from 'react';

const Image_Uploader = () => {
	const [img, setImg] = useState(null);

	const imageUploader = async (val: any) => {
		const _my_image = val.target.files[0];
		setImg(_my_image);
	};
	const btnClick = () => {
		console.log(img);
		console.log('button clicked');
		// const data = new FormData();
		// data.append('my_img', img);

		// Send a POST request
		//    axios({
		//      method: "post",
		//      url: "http://localhost:5000/ram",
		//      data,
		//      header: {
		//        "Content-Type": "multipart/form-data",
		//      },
		//    })
		//      .then((response) => {
		//        console.log("image uploaded in backend");
		//        console.log(response);
		//        console.log("bye bye");
		//      })
		//      .catch((err) => {
		//        console.log("ho gaya satyanash");
		//        console.log(err);
		//      });
	};

	return (
		<div className="bg-pink">
			<div className="flex flex-col items-center justify-center">
				<div>
					{img ? (
						<Image
							className="border-2 border-black rounded"
							src={URL.createObjectURL(img)}
							height={300}
							width={300}
							alt="Image"
						/>
					) : (
						<Image
							className="border-2 border-black rounded-xl bg-white"
							src={'/upload_image_icon.png'}
							height={300}
							width={300}
							alt="Image"
						/>
					)}
				</div>
				<input
					className="justify-center"
					type="file"
					accept="image/png, image/jpeg"
					onChange={imageUploader}
				/>
			</div>
			
		</div>
	);
};

export default Image_Uploader;
